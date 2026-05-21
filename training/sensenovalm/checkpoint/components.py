# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import copy
import os
import re
from collections import defaultdict

import torch
from torch.distributed._shard.api import load_with_process_group

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer import TrainState
from sensenovalm.model.moe.moe import MoEBase
from sensenovalm.solver.optimizer import HybridZeroOptimizer, HybridZeroOptimizer_v2
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.parallel import is_using_isp
from sensenovalm.utils.storage_manager import get_fns, llm_load, llm_save

from .utils import (
    get_model_topology,
    get_non_moe_state_dict,
    get_shard_state_dict,
    load_shard_state_dict,
    obtain_not_saved_mtp_state,
)

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


def _has_mot_resume_weights(states):
    """Detect whether checkpoint states already include MoT generation-branch language weights."""
    if "language_model.layers.0.attention.wq_mot_gen.weight" in states:
        return True
    if "model.layers.0.self_attn.q_proj_mot_gen.weight" in states:
        return True

    for key in states.keys():
        if "mot_gen" not in key:
            continue
        if key.startswith("language_model.layers.") or key.startswith("model.layers."):
            return True
    return False


# only support auto resume
def try_load_moe_checkpoint(folder, model, state_dict, expert_mp_rank, pp_rank):
    """Load MoE layer parameters from separate files if the model has MoE layers."""
    # Calculate the stage size and rank within the pipeline parallelism
    pp_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pp_stage_size
    moe_layer_id += gpc.config.model.moe_kwargs.get("first_k_dense_replace", 0)
    mode = "wp" if is_using_isp() else "tp"
    load_mtp_params = gpc.config.get("load_mtp_params", True)
    mot_model = getattr(gpc.config.model, "mot_model", False)
    fns = get_fns(folder) or []
    moe_file_pattern = re.compile(rf"^model_moe_layer(\d+)_expert\d+_{mode}{expert_mp_rank}\.pt$")
    available_moe_layers = set()
    available_moe_files = set()
    for fn in fns:
        if fn.endswith(".md5"):
            continue
        matched = moe_file_pattern.match(fn)
        if not matched:
            continue
        available_moe_layers.add(int(matched.group(1)))
        available_moe_files.add(fn)

    if len(available_moe_layers) == 0:
        return
    max_available_moe_layer = max(available_moe_layers)

    # Iterate over all MoE modules in model order. Do not cap by `num_layers` here:
    # in MoT models, feed_forward and feed_forward_mot_gen both have MoE layers.
    for _, module in model.named_modules():
        if not isinstance(module, MoEBase):
            continue

        # The source checkpoint might only contain one branch (e.g. und only),
        # while current MoT model has extra MoE modules for generation branch.
        # Stop loading once we exceed the max serialized MoE layer id.
        if moe_layer_id > max_available_moe_layer:
            if gpc.is_rank_for_log():
                logger.warning(
                    f"Stop loading MoE shards at layer {moe_layer_id}, "
                    f"because checkpoint max MoE layer is {max_available_moe_layer}."
                )
            break

        # The legacy MTP stop condition is based on `num_layers`, which is incompatible
        # with MoT checkpoints where MoE shards are serialized for both branches.
        if not load_mtp_params and not mot_model:
            if (
                "language_model.layers.0.attention.wk_hw.weight" not in state_dict
                and moe_layer_id >= (gpc.config.model.num_layers - gpc.config.model.extra_num_layers)
            ):
                break
            if (
                "language_model.layers.0.attention.wk_hw.weight" in state_dict
                and moe_layer_id > gpc.config.model.num_layers
            ):
                break
        num_local_wrapped_experts = len(module.moe_layer.experts.wrapped_experts)
        expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)
        # loop all local_experts
        for local_expert_id in range(num_local_wrapped_experts):
            global_expert_id = expp_rank * num_local_wrapped_experts + local_expert_id
            fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_{mode}{expert_mp_rank}.pt"
            assert (
                fn in available_moe_files
            ), f"{os.path.join(folder, fn)} is not found! existing max moe layer id is {max_available_moe_layer}"
            fp = os.path.join(folder, fn)
            expert_state_dict = llm_load(fp, map_location=get_current_device())
            # Updating global -> local expert ids
            moe_str_prefix = ".moe_layer.experts.wrapped_experts."
            for key in list(expert_state_dict.keys()):
                # dict_keys(['w1', 'w3', 'w2']
                # layers.41.feed_forward.moe_layer.experts.wrapped_experts.0.w1.weight
                # local_key = f"layers.{moe_layer_id}.feed_forward.moe_layer.experts.wrapped_experts.0.{key}.weight"
                local_key = key.replace(f"{moe_str_prefix}{global_expert_id}", f"{moe_str_prefix}{local_expert_id}")
                expert_state_dict[local_key] = expert_state_dict.pop(key)
            state_dict.update(expert_state_dict)
        moe_layer_id += 1


def try_save_moe_checkpoint(folder, model, expert_mp_rank, pp_rank, state_dict_override=None):
    # Using layer_#_expert_# to save the model's expert state_dict，a hack.
    pipeline_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pipeline_stage_size
    moe_layer_id += gpc.config.model.moe_kwargs.get("first_k_dense_replace", 0)
    mode = "wp" if is_using_isp() else "tp"
    for n_module, module in model.named_modules():
        if isinstance(module, MoEBase):
            num_local_wrapped_experts = len(module.moe_layer.experts.wrapped_experts)
            expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)

            # get all moe parameters
            moe_state_dict = {}
            if state_dict_override is None:
                for n, p in module.state_dict().items():
                    if "expert" in n and "moe_layer.gate" not in n:
                        moe_state_dict[n_module + "." + n] = p
            else:
                prefix = f"{n_module}." if n_module else ""
                for key, tensor in state_dict_override.items():
                    if not key.startswith(prefix):
                        continue
                    local_key = key[len(prefix) :]
                    if "expert" in local_key and "moe_layer.gate" not in local_key:
                        moe_state_dict[key] = tensor
            moe_str_prefix = ".moe_layer.experts.wrapped_experts."
            # Reorder the moe name rank, so that each checkpoint only has one expert
            experts_state_dict = defaultdict(dict)
            for key in list(moe_state_dict.keys()):
                m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)

                local_expert_id = None
                if not m:
                    logger.warning(f"No expert found in key {key}.")
                else:
                    local_expert_id = m.group(1)

                global_expert_id = expp_rank * num_local_wrapped_experts + int(local_expert_id)
                expert_key = key.replace(f"{moe_str_prefix}{local_expert_id}", f"{moe_str_prefix}{global_expert_id}")

                # truncating extra tensor (shared) storage
                truncated = moe_state_dict.pop(key).clone().detach()
                experts_state_dict[str(global_expert_id)][expert_key] = truncated

            # let save the moe parameters
            for global_expert_id, expert_state_dict in experts_state_dict.items():
                # save the moe parameters
                fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_{mode}{expert_mp_rank}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=expert_state_dict)
            moe_layer_id += 1


# def load_model_checkpoint(folder, model):
def _load_rank_model_states(folder, model):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If fsdp is activated, the saved weight is named:
    - folder
        - model_tp{tp_rank}_pp{pp_rank}_zo{zo_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.

    Shared loader for both `load_model_checkpoint` and `load_model_state_dict`.
    It loads the current-rank shard file and merges MoE expert shards (if any).

    """

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    dp_size = gpc.get_world_size(ParallelMode.DATA)

    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)

    fns = get_fns(folder)

    # avoid ckpt misuse between FSDP and no-FSDP
    _start_with = "model_w" if is_using_isp() else "model_t"
    test_fn = list([f for f in fns if f.startswith(_start_with) and not f.endswith(".md5")]).pop()
    assert ("_dp" in test_fn and gpc.config.parallel.zero1.fsdp) or (
        "_dp" not in test_fn and not gpc.config.parallel.zero1.fsdp
    ), "FSDP model wants to load no-FSDP ckpts or reverse"

    max_pp, max_wp, max_tp, max_zo = 0, 0, 0, 0
    for fn in fns:
        if fn.startswith(_start_with) and not fn.endswith(".md5"):
            segements = os.path.splitext(fn)[0].split("_")
            if is_using_isp():
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_wp = max(max_wp, int(segements[-2][2:]))
            elif gpc.config.parallel.zero1.fsdp:
                max_zo = max(max_zo, int(segements[-1][2:]))
                max_pp = max(max_pp, int(segements[-2][2:]))
                max_tp = max(max_tp, int(segements[-3][2:]))
            else:
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_tp = max(max_tp, int(segements[-2][2:]))

    assert (
        pp_size == max_pp + 1
    ), f"The weights are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    assert (
        wp_size == max_wp + 1
    ), f"The weights are save for {max_wp+1} parallelism, while current has {wp_size} weight parallelism"
    if not is_using_isp():
        assert (
            tp_size == max_tp + 1
        ), f"The weights are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"
    if gpc.config.parallel.zero1.fsdp:
        assert (
            dp_size == max_zo + 1
        ), f"The weights are save for {max_zo+1} FSDP shards , while current has {dp_size} FSDP shards"

    if is_using_isp():
        should_load_name = f"model_wp{wp_rank}_pp{pp_rank}.pt"
    elif gpc.config.parallel.zero1.fsdp:
        should_load_name = f"model_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt"
    else:
        should_load_name = f"model_tp{tp_rank}_pp{pp_rank}.pt"
    fp = os.path.join(folder, should_load_name)

    # for FSDP shards loading, we need to set process group
    with load_with_process_group(gpc.get_group(ParallelMode.ZERO1)):
        states = llm_load(fp, map_location=get_current_device())

    # Merge MoE expert weights if needed
    if gpc.config.model.use_moe:
        if is_using_isp():
            expert_wp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
            try_load_moe_checkpoint(folder, model, states, expert_wp_rank, pp_rank)
        else:
            expert_tp_rank = 0 if gpc.config.parallel.expert.no_tp else tp_rank
            try_load_moe_checkpoint(folder, model, states, expert_tp_rank, pp_rank)
    
    model_config = gpc.config.model
    mot_model = getattr(model_config, 'mot_model', False)
    mot_random_init = getattr(model_config, 'mot_random_init', True)
    if mot_model:
        if _has_mot_resume_weights(states):
            # 续训
            logger.info('Initializing MoT VLM from one pre-trained MoT VLM')
            logger.info("Need to check if has missing keys or unexpected keys when MoT continuel training")
        else:
            if mot_random_init:
                logger.info("Initializing the generation branch from the scrach")
                logger.info("norm the qk with one")
                if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                    norm_weight = states["language_model.layers.0.attention.q_norm.weight"]
                    for j in range(0, model_config.num_layers_for_pp):
                        for name in ["q", "k"]:
                            states[f"language_model.layers.{j}.attention.{name}_norm_mot_gen.weight"] = torch.ones(
                                norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype
                            )
                            states[f"language_model.layers.{j}.attention.{name}_norm_hw_mot_gen.weight"] = torch.ones(
                                norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype
                            )
            else:
                logger.info('Initializing the generation branch from the pre-trained Dense VLM')
                # mlp 
                states_keys = list(states.keys())
                for name in states_keys:
                    if "feed_forward" in name:
                        states[name.replace("feed_forward", "feed_forward_mot_gen")] = states[name]

                for i in range(0, model_config.num_layers_for_pp):
                    # norm
                    states[f"language_model.layers.{i}.ffn_norm_mot_gen.weight"] = states[f"language_model.layers.{i}.ffn_norm.weight"]
                    states[f"language_model.layers.{i}.attention_norm_mot_gen.weight"] = states[f"language_model.layers.{i}.attention_norm.weight"]

                    # attention
                    states[f"language_model.layers.{i}.attention.wq_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wq.weight"]
                    states[f"language_model.layers.{i}.attention.wq_hw_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wq_hw.weight"]
                    states[f"language_model.layers.{i}.attention.wk_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wk.weight"] 
                    states[f"language_model.layers.{i}.attention.wk_hw_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wk_hw.weight"]
                    states[f"language_model.layers.{i}.attention.wv_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wv.weight"]
                    states[f"language_model.layers.{i}.attention.wo_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.wo.weight"]

                    # qk_norm
                    if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                        states[f"language_model.layers.{i}.attention.q_norm_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.q_norm.weight"]
                        states[f"language_model.layers.{i}.attention.q_norm_hw_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.q_norm_hw.weight"]
                        states[f"language_model.layers.{i}.attention.k_norm_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.k_norm.weight"]
                        states[f"language_model.layers.{i}.attention.k_norm_hw_mot_gen.weight"] = states[f"language_model.layers.{i}.attention.k_norm_hw.weight"]

    else:
        logger.info("Not Using Correct Repo for training!!!")
        if 'language_model.layers.0.attention.wk_hw.weight' not in states:
            # NOTE: rope model
            language_model_state_dict = dict()

            # feed forward layer / moe 
            states_keys = list(states.keys())
            for name in states_keys:
                split_names = name.split(".")
                if "feed_forward" in split_names:
                    layer_num = int(split_names[2])
                    split_names[2] = str(layer_num - model_config.first_layer + model_config.extra_num_layers)
                    new_name = '.'.join(split_names)
                    language_model_state_dict[new_name] = states.pop(name)
                    if layer_num < model_config.extra_num_layers:
                        if gpc.config.init_buffer_method == "random":
                            pass
                        elif gpc.config.init_buffer_method == "copy":
                            language_model_state_dict[name] = language_model_state_dict[new_name].clone()

            for i in range(0, model_config.num_layers - model_config.extra_num_layers):
                # norm
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.ffn_norm.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.ffn_norm.weight"
                )
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention_norm.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.attention_norm.weight"
                )
                # attention
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wq.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.attention.wq.weight"
                )
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wk.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.attention.wk.weight"
                )
                    
                # NOTE: rope model
                # TODO: support interleaved rope embedding, now only support half and half
                def initialize_equal_weight_for_HW(model_weight, head_dim):
                    model_weight_t = model_weight.permute(1, 0)
                    num_head = model_weight_t.shape[1] // head_dim
                    model_reshape = model_weight_t.reshape(model_weight_t.shape[0], num_head, head_dim)
                    model_w1, _, model_w3, _ = model_reshape.chunk(4, dim=-1)
                    model_weight_hw = torch.cat([model_w1, model_w3], dim=-1).repeat(1, 1, 2).reshape(model_weight_t.shape[0], -1)
                    return model_weight_hw.permute(1, 0)
                        
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wq_hw.weight"] = \
                    initialize_equal_weight_for_HW(language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wq.weight"], model_config.head_dim)

                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wk_hw.weight"] = \
                    torch.zeros_like(language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wk.weight"])

                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wv.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.attention.wv.weight"
                )
                language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.wo.weight"] = states.pop(
                    f"language_model.layers.{i+model_config.first_layer}.attention.wo.weight"
                )

                if gpc.config.model_type == "QWEN3_MoE":
                    for name in ["q", "k"]:
                        norm_weight = states.pop(
                            f"language_model.layers.{i + model_config.first_layer}.attention.{name}_norm.weight"
                        )
                        language_model_state_dict[f"language_model.layers.{i + model_config.extra_num_layers}.attention.{name}_norm.weight"] = norm_weight
                        
                        language_model_state_dict[f"language_model.layers.{i+model_config.extra_num_layers}.attention.{name}_norm_hw.weight"] = \
                            torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)
            
            # NOTE: postlayer ---- #
            for j in range(model_config.extra_num_layers):
                if gpc.config.model_type == "QWEN3_MoE":
                    # qk norm
                    for name in ["q", "k"]:
                        language_model_state_dict[f"language_model.layers.{j}.attention.{name}_norm.weight"] = \
                            torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)

                        language_model_state_dict[f"language_model.layers.{j}.attention.{name}_norm_hw.weight"] = \
                            torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)
                if gpc.config.init_buffer_method == "random":
                    pass
                elif gpc.config.init_buffer_method == "copy":
                    # norm
                    language_model_state_dict[f"language_model.layers.{j}.ffn_norm.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.ffn_norm.weight"].clone()
                    language_model_state_dict[f"language_model.layers.{j}.attention_norm.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention_norm.weight"].clone()
                    # attention
                    language_model_state_dict[f"language_model.layers.{j}.attention.wq.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wq.weight"].clone()
                    language_model_state_dict[f"language_model.layers.{j}.attention.wk.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wk.weight"].clone()

                    language_model_state_dict[f"language_model.layers.{j}.attention.wq_hw.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wq_hw.weight"].clone()   
                    language_model_state_dict[f"language_model.layers.{j}.attention.wk_hw.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wk_hw.weight"].clone()
                    language_model_state_dict[f"language_model.layers.{j}.attention.wv.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wv.weight"].clone()
                    language_model_state_dict[f"language_model.layers.{j}.attention.wo.weight"] = language_model_state_dict[f"language_model.layers.{j+model_config.extra_num_layers}.attention.wo.weight"].clone()

            
            if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
                language_model_state_dict["language_model.tok_embeddings.weight"] = states.pop("language_model.tok_embeddings.weight")

            if (
                os.environ.get("lm_head_no_split", "null").lower() == "true"
                or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
            ):
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    language_model_state_dict["language_model.output.weight"] = states.pop("language_model.output.weight")
                    language_model_state_dict["language_model.norm.weight"] = states["language_model.norm.weight"]
            else:
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    language_model_state_dict["language_model.output.weight"] = states.pop("language_model.output.weight")
                    language_model_state_dict["language_model.norm.weight"] = states["language_model.norm.weight"]
            
            states = language_model_state_dict

        else:
            print(f"TODO: need to check here when starting stage2 !")



    _maybe_fill_mtp_states(model, states)

    return states


def _maybe_fill_mtp_states(model, states):
    expected_keys = set(model.state_dict().keys())
    missing_keys = [k for k in expected_keys if k not in states]
    if missing_keys:
        mtp_missing_state, _ = obtain_not_saved_mtp_state(states, missing_keys)
        states.update(mtp_missing_state)


def load_model_checkpoint(folder, model):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If fsdp is activated, the saved weight is named:
    - folder
        - model_tp{tp_rank}_pp{pp_rank}_zo{zo_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.
    """

    states = _load_rank_model_states(folder, model)

    """
    # need convert the gate parameters to float32 (to fit deepspeed style mechanism), it may cause round-off in
    # gate.weight. The conversion will also be done when doing forward. so we can just comment it out. this make
    # the gate parameters to be float16 before forward.
    for key in list(states.keys()):
        if 'moe_layer.gate.wg.weight' in key:
            states[key] = states[key].float()
            print("load: ", states[key].float(),flush=True)
    """




    if gpc.config.parallel.zero1.fsdp:
        missing_k, unexpected_keys = load_shard_state_dict(model, states, strict=False)
    else:
        missing_k, unexpected_keys = model.load_state_dict(states, strict=False)
    if len(missing_k) != 0:
        logger.warning(f"Warning: missing keys {missing_k}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Warning: unexpected keys {unexpected_keys}")

    # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
    del states
    sensenovalm_accelerator.empty_cache()

def load_model_state_dict(folder, model):
    """
    Load the checkpoint shard(s) into a model-like state_dict for the current rank, without applying it to `model`.
    This is mainly used to resume EMA/SWA averaged model weights from `<ckpt>/averaged_model/`
    while keeping training weights untouched.
    It follows the same sharding / naming rules as `load_model_checkpoint`.
    """

    states = _load_rank_model_states(folder, model)
    return states

# def save_model_checkpoint(folder, model):
def save_model_checkpoint(folder, model, state_dict_override=None):
    
    """
    Save the model according to the relationship between tp and dp. The principle is that the data of each tp
    will not be gathered and saved separately, which is equivalent to actual sharding. The saved weight is named
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If fsdp is activated, the saved weight is named:
    - folder
        - model_tp{tp_rank}_pp{pp_rank}_zo{zo_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.

    Args:
        folder: The folder to save the model
        model: The model to be saved
    """

    full_states = None
    if state_dict_override is not None:
        full_states = state_dict_override
        # Preserve expert tensors for the separate MoE checkpoint writer.
        states = dict(state_dict_override)
    else:
        if gpc.config.parallel.zero1.fsdp:
            states = get_shard_state_dict(model)
        else:
            states = model.state_dict()

    # get non-expert parameters
    states = get_non_moe_state_dict(states)
    topo = get_model_topology(model)

    if folder is not None:
        dp_size = gpc.get_world_size(ParallelMode.DATA)
        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
        pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        wdp_rank = gpc.get_local_rank(ParallelMode.WEIGHT_DATA)

        should_save_rank_pair = set()  # (tp_rank, dp_rank)

        # TODO In theory, we should also consider pp level, but since pp is generally a state across machines,
        # even if pp is not considered, it will definitely not be written on the same machine.

        # for tensor parallel mode with isp
        if is_using_isp():
            if wdp_rank == 0:
                fn = f"model_wp{wp_rank}_pp{pp_rank}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=states)
                topo_fn = f"topo_wp{wp_rank}_pp{pp_rank}.json"
                topo_fp = os.path.join(folder, topo_fn)
                llm_save(topo_fp, saved_obj=topo)
            if gpc.config.model.use_moe:
                expert_wp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
                expert_wdp_rank = gpc.get_local_rank(ParallelMode.EXPERT_DATA)
                if expert_wdp_rank == 0:
                    try_save_moe_checkpoint(
                        folder,
                        model,
                        expert_wp_rank,
                        pp_rank,
                        state_dict_override=full_states,
                    )
        else:
            # for tensor parallel mode with mtp/msp/fsp
            for i in range(tp_size):
                if gpc.config.parallel.zero1.fsdp:
                    for j in range(dp_size):
                        should_save_rank_pair.add((i, j))
                else:
                    should_save_rank_pair.add((i, i % dp_size))

                if (tp_rank, dp_rank) in should_save_rank_pair:
                    f_dp = f"_dp{dp_rank}" if gpc.config.parallel.zero1.fsdp else ""
                    fn = f"model_tp{tp_rank}_pp{pp_rank}{f_dp}.pt"
                    fp = os.path.join(folder, fn)
                    llm_save(fp, saved_obj=states)
                    if not gpc.config.parallel.zero1.fsdp or dp_rank == tp_rank % dp_size:
                        topo_fn = f"topo_tp{tp_rank}_pp{pp_rank}.json"
                        topo_fp = os.path.join(folder, topo_fn)
                        llm_save(topo_fp, saved_obj=topo)

            # try to save expert parameter to separate files if model have moe layer
            if gpc.config.model.use_moe:
                expert_dp_size = gpc.get_world_size(ParallelMode.EXPERT_DATA)
                expert_tp_size = 1 if gpc.config.parallel.expert.no_tp else tp_size
                expert_dp_rank = gpc.get_local_rank(ParallelMode.EXPERT_DATA)
                expert_tp_rank = 0 if gpc.config.parallel.expert.no_tp else tp_rank
                should_save_rank_pair.clear()
                for i in range(expert_tp_size):
                    should_save_rank_pair.add((i, i % expert_dp_size))
                if (expert_tp_rank, expert_dp_rank) in should_save_rank_pair:
                    try_save_moe_checkpoint(
                        folder,
                        model,
                        expert_tp_rank,
                        pp_rank,
                        state_dict_override=full_states,
                    )

    torch.distributed.barrier()


def load_optimizer_checkpoint(folder, optim):
    """Load the optimizer state from the local file system or remote
    object storage Service (OSS).

    Args:
        optim (Optimizer): optimizer
        folder (str): The FS/OSS path where the optimizer will be stored.
    """

    fns = get_fns(folder)
    max_tp, max_wp, max_pp, max_zero = 0, 0, 0, 0
    use_moe = gpc.config.model.use_moe
    max_ep, max_ewp, max_moe_zero = 0, 0, 0
    is_moe_optim = False
    for fn in fns:
        if fn.startswith("optimizer_") and not fn.endswith(".md5"):
            if is_using_isp():
                if fn.startswith("optimizer_ep"):
                    is_moe_optim = True
                    _, ep, ewp, pp, moe_zero = os.path.splitext(fn)[0].split("_")
                else:
                    _, wp, pp, zero = os.path.splitext(fn)[0].split("_")
                max_pp = max(max_pp, int(pp[2:]))
                if is_moe_optim:
                    max_ep = max(max_ep, int(ep[2:]))
                    max_ewp = max(max_ewp, int(ewp[3:]))
                    max_moe_zero = max(max_moe_zero, int(moe_zero[2:]))
                else:
                    max_zero = max(max_zero, int(zero[2:]))
                    max_wp = max(max_wp, int(wp[2:]))
            else:
                if fn.startswith("optimizer_ep"):
                    is_moe_optim = True
                    _, ep, tp, pp, moe_zero = os.path.splitext(fn)[0].split("_")
                else:
                    _, tp, pp, zero = os.path.splitext(fn)[0].split("_")
                max_tp = max(max_tp, int(tp[2:]))
                max_pp = max(max_pp, int(pp[2:]))
                if is_moe_optim:
                    max_moe_zero = max(max_moe_zero, int(moe_zero[2:]))
                    max_ep = max(max_ep, int(ep[2:]))
                else:
                    max_zero = max(max_zero, int(zero[2:]))

    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    moe_zero_size = gpc.get_world_size(ParallelMode.EXPERT_ZERO1)
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)
    moe_zero_rank = gpc.get_local_rank(ParallelMode.EXPERT_ZERO1)
    if is_using_isp():
        ewp_size = gpc.get_world_size(ParallelMode.EXPERT_WEIGHT)
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)

    assert (
        pp_size == max_pp + 1
    ), f"The optimizer states are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    if not is_using_isp():
        assert (
            tp_size == max_tp + 1
        ), f"The optimizer states are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"
    assert (
        wp_size == max_wp + 1
    ), f"The optimizer states are save for {max_wp+1} parallelism, while current has {wp_size} weight parallelism"
    assert zero_size == max_zero + 1, (
        f"The optimizer states are save for {max_zero+1} zero parallel, "
        f"while current has {zero_size} zero broadcast range."
    )
    if is_moe_optim:
        assert moe_zero_size == max_moe_zero + 1, (
            f"The optimizer states are save for {max_moe_zero+1} expert zero parallelism, "
            f"while current has {moe_zero_size} expert zero broadcast range."
        )
        assert (
            ep_size == max_ep + 1
        ), f"The optimizer states are save for {max_ep+1} parallelism, while current has {ep_size} weight parallelism"
        if is_using_isp():
            assert ewp_size == max_ewp + 1, (
                f"The optimizer states are save for {max_ewp+1} expert weight parallelism,"
                f" while current has {ewp_size} expert weight parallelism"
            )

    if is_using_isp():
        if use_moe and moe_zero_size * ep_size * ewp_size > zero_size * wp_size:
            fp = f"optimizer_ep{ep_rank}_ewp{ewp_rank}_pp{pp_rank}_zo{moe_zero_rank}.pt"
        else:
            fp = f"optimizer_wp{wp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
    else:
        if use_moe and moe_zero_size * ep_size > zero_size:
            fp = f"optimizer_ep{ep_rank}_tp{tp_rank}_pp{pp_rank}_zo{moe_zero_rank}.pt"
        else:
            fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"

    states = llm_load(os.path.join(folder, fp), map_location=get_current_device())

    if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        fp_meta = os.path.join(folder, optim.rank_unique_id)
        try:
            zero_devide_optim_plan = llm_load(fp_meta)
            states.update({"zero_devide_optim_plan": zero_devide_optim_plan})
        except Exception as e:
            if gpc.is_rank_for_log():
                logger.warning(
                    f"Read zero optimzer split file '{fp_meta}', for '{e}'"
                    f"Please check whether loading ckpts are saved with the HybridZeroOptimizer."
                )

    # compatible with old code that only have one param group, need to align with both parameter groups
    if len(states["base_optim_states"]["param_groups"]) == 1:
        for group in optim.param_groups:
            # for new added empty group, since it has no params, just create it fakely
            if len(group["params"]) == 0:
                states["base_optim_states"]["param_groups"].append(group)
            # for origin group, create new added attributes in recent updates
            else:
                saved_group = states["base_optim_states"]["param_groups"][0]
                saved_group["dp_mode"] = group["dp_mode"]
                saved_group["dtype"] = group["dtype"]

    optim.load_state_dict(states)
    del states
    sensenovalm_accelerator.empty_cache()


def save_optimizer_checkpoint(optim, state_path):
    """Store the state of the optimizer to the local file system or remote OSS.

    Args:
        optim (Optimizer)
        state_path (str): The state loading path of optimizer.
    """

    # TODO sanity check for optimizer type
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    dp_size = gpc.get_world_size(ParallelMode.DATA)
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)
    moe_zero_rank = gpc.get_local_rank(ParallelMode.EXPERT_ZERO1)
    moe_data_size = gpc.get_world_size(ParallelMode.EXPERT_DATA)
    moe_zero_size = gpc.get_world_size(ParallelMode.EXPERT_ZERO1)
    if is_using_isp():
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
        ewp_size = gpc.get_world_size(ParallelMode.EXPERT_WEIGHT)

    use_moe = gpc.config.model.use_moe

    states = optim.state_dict()

    if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        if is_using_isp():
            if use_moe and moe_zero_size * ep_size * ewp_size > zero_size * wp_size:
                fp = f"optimizer_ep{ep_rank}_ewp{ewp_rank}_pp{pp_rank}_zo{moe_zero_rank}.pt"
                if (gpc.get_global_rank() % (ewp_size * ep_size * moe_data_size)) < moe_zero_size * ewp_size * ep_size:
                    llm_save(os.path.join(state_path, fp), states)
            else:
                fp = f"optimizer_wp{wp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
                if (gpc.get_global_rank() % (tp_size * dp_size)) < zero_size * wp_size:
                    llm_save(os.path.join(state_path, fp), states)
        else:
            if use_moe and moe_zero_size * ep_size > zero_size:
                fp = f"optimizer_ep{ep_rank}_tp{tp_rank}_pp{pp_rank}_zo{moe_zero_rank}.pt"
                if (gpc.get_global_rank() % (tp_size * ep_size * moe_data_size)) < moe_zero_size * tp_size * ep_size:
                    llm_save(os.path.join(state_path, fp), states)
            else:
                fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
                if (gpc.get_global_rank() % (tp_size * dp_size)) < zero_size * tp_size:
                    llm_save(os.path.join(state_path, fp), states)
        if "zero_devide_optim_plan" in states:
            params_per_rank_id_dict = states.pop("zero_devide_optim_plan")
            fp_meta = os.path.join(state_path, optim.rank_unique_id)
            llm_save(fp_meta, params_per_rank_id_dict)
    else:
        llm_save(os.path.join(state_path, fp), states)


def load_sampler(ckpt_path: str, sampler):
    sampler_states = llm_load(os.path.join(ckpt_path, "sampler.pt"))
    sampler.load_state_dict(sampler_states)
    if gpc.is_rank_for_log():
        pstate = copy.deepcopy(sampler_states)
        pstate.pop("indices", None)
        pstate.pop("rng_state", None)
        logger.info(f"reload sampler_states:{pstate}")
    sensenovalm_accelerator.empty_cache()


def load_context(ckpt_path: str, train_state: TrainState):
    context_stuffs = llm_load(os.path.join(ckpt_path, "context.pt"))
    train_state.load_state_dict(context_stuffs)
    if gpc.is_rank_for_log():
        logger.info(f"reload train_state:{train_state}")
    sensenovalm_accelerator.empty_cache()


def load_scheduler(ckpt_path: str, lr_scheduler, optimizer, train_state: TrainState):
    learning_rate = train_state.lr
    scheduler_states = llm_load(os.path.join(ckpt_path, "schedulder.pt"))
    
    if learning_rate != scheduler_states["base_lrs"][0] and gpc.is_rank_for_log():
        logger.warning(
            f"Using new learning rate {learning_rate} to replace old learn rate {scheduler_states['base_lrs'][0]}."
        )

    base_lrs = copy.deepcopy(scheduler_states["base_lrs"])
    scheduler_states["base_lrs"] = [learning_rate] * len(scheduler_states["base_lrs"])
    if "after_scheduler_dict" in scheduler_states:
        scheduler_states["after_scheduler_dict"]["base_lrs"] = [learning_rate] * len(
            scheduler_states["after_scheduler_dict"]["base_lrs"]
        )

    lr_scheduler.load_state_dict(scheduler_states)

    lr_scheduler_offset = int(gpc.config.get('lr_scheduler_offset', 0))
    # step_count have been updated before saving checkpoint.
    lr_scheduler.last_epoch = max(train_state.step_count - lr_scheduler_offset, 0)
    lr_scheduler._step_count = lr_scheduler.last_epoch + 1

    # compatible with old code that only have one param group
    if len(base_lrs) == 1:
        base_lrs = base_lrs * len(optimizer.param_groups)

    ratios = [learning_rate / lr for lr in base_lrs]
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = param_group["lr"] * ratios[idx]
    sensenovalm_accelerator.empty_cache()

    if gpc.is_rank_for_log():
        logger.info(f"reload load_scheduler:{lr_scheduler}")
