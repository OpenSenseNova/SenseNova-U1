# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import json
import logging
import os
import re
import shutil
import time

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import (  # shard_checkpoint,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.storage_manager import get_fns, llm_load

shard_checkpoint = llm_load

# global llm logger
logger = logging.getLogger(__file__)
sensenovalm_accelerator = get_accelerator()


def copy_files(src_dir, dst_dir):
    """
    copy files from src_dir to dst_dir.
    """
    # check src dir
    assert os.path.exists(src_dir), f"src dir: {src_dir} is not exists!"

    # create dst dir
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)


def _find_max_wp_pp(names):
    ckpt_names = []
    for name in names:
        if name.startswith("model_w") and not name.endswith("md5"):
            ckpt_names.append(name)

    max_wp, max_pp = -1, -1
    for ckpt in ckpt_names:
        _, wp, pp = os.path.splitext(ckpt)[0].split("_")
        max_wp = max(max_wp, int(wp[2:]) + 1)
        max_pp = max(max_pp, int(pp[2:]) + 1)

    return max_wp, max_pp


def load_source_ckpt(src):
    """
    load model_config.pt, and model_wp{x}_pp{x}.pt(not including moe currently) from ``src``

    :return:
        - model_config: dict
        - states: 2-d array. states[i][j] stands for state_dict of wp_i pp_j
    """

    # check wp pp size
    ckpt_names = get_fns(src)
    max_wp, max_pp = _find_max_wp_pp(ckpt_names)
    assert max_pp <= 1, "pipeline parallel is not supported!"

    # 2-d array wp_rank, pp_rank
    # print("Source Checkpoint Loading", flush=True)
    states = [[None for _ in range(max_pp)] for __ in range(max_wp)]
    for wp in tqdm(range(max_wp)):
        for pp in tqdm(range(max_pp)):
            ckpt_name = os.path.join(src, f"model_wp{wp}_pp{pp}.pt")
            states[wp][pp] = llm_load(ckpt_name, map_location="cpu")

    return states


def load_source_moe_ckpt(src):
    ckpt_names = get_fns(src)
    moe_ckpts = {}
    mode = None
    for name in ckpt_names:
        if name.startswith("model_moe_layer") and name.endswith("pt"):
            if mode is None:
                mode = "tp" if "_tp" in name else "wp"
            matched = re.match(f"model_moe_layer([0-9]+)_expert([0-9]+)_{mode}([0-9]+).pt", name)
            layer_id, expert_i, mode_rank = int(matched.group(1)), int(matched.group(2)), int(matched.group(3))
            if moe_ckpts.get(layer_id) is None:
                moe_ckpts[layer_id] = {}
            if moe_ckpts[layer_id].get(expert_i) is None:
                moe_ckpts[layer_id][expert_i] = []
            ckpt_name = os.path.join(src, name)
            moe_ckpts[layer_id][expert_i].append((mode_rank, llm_load(ckpt_name, map_location="cpu")))

    for _, moe_layer_ckpts in moe_ckpts.items():
        for expert_i in moe_layer_ckpts:
            moe_layer_ckpts[expert_i] = [ckpt[1] for ckpt in sorted(moe_layer_ckpts[expert_i], key=lambda x: x[0])]

    convert_moe_ckpts = convert_moe_ckpt(moe_ckpts)

    return convert_moe_ckpts


def convert_moe_ckpt(moe_ckpts):
    merge_dim0 = False
    use_gemm = True
    model_args = gpc.config.model
    moe_type = model_args.moe_kwargs.moe_type
    num_experts = model_args.moe_kwargs.num_experts
    ep_size = gpc.config.parallel["expert"]["size"]
    tp_mode = gpc.config.parallel["tensor"]["mode"]
    num_local_experts = num_experts // ep_size
    if moe_type == "GShard" and model_args.moe_layer_kwargs.get("use_grouped_mlp", True):
        use_gemm = False

    if (use_gemm and tp_mode == "isp") or moe_type == "MegaBlock-D":
        merge_dim0 = True

    # merge tp/wp
    if merge_dim0:
        col_dim = 0
        row_dim = 0
    elif use_gemm:  # (num_local_experts, input_feature, output_feature)
        col_dim = 2
        row_dim = 1
    else:  # (output_feature, input_feature)
        if tp_mode == "isp":
            col_dim = 0
            row_dim = 0
        else:
            col_dim = 0
            row_dim = 1

    w_pattern = r"\.(w\d)\.weight|\.(w\d)$"
    fc_pattern = r"\.(fc\d)\.weight|\.(fc\d)\.bias|\.(fc\d)$"

    merged_moe_ckpts = {}
    for layer_id in moe_ckpts:
        if merged_moe_ckpts.get(layer_id) is None:
            merged_moe_ckpts[layer_id] = {}
        for expert_i in moe_ckpts[layer_id]:
            if merged_moe_ckpts[layer_id].get(expert_i) is None:
                merged_moe_ckpts[layer_id][expert_i] = {}
            cur_ckpts = moe_ckpts[layer_id][expert_i]
            for key in cur_ckpts[0].keys():
                if re.search(w_pattern, key):
                    w_i = re.search(w_pattern, key).group(1) or re.search(w_pattern, key).group(2)
                    if w_i != "w2":
                        merged_moe_ckpts[layer_id][expert_i][key] = torch.cat(
                            [ckpt[key] for ckpt in cur_ckpts], dim=col_dim
                        )
                    else:
                        merged_moe_ckpts[layer_id][expert_i][key] = torch.cat(
                            [ckpt[key] for ckpt in cur_ckpts], dim=row_dim
                        )
                elif re.search(fc_pattern, key):
                    fc_i = (
                        re.search(fc_pattern, key).group(1)
                        or re.search(fc_pattern, key).group(2)
                        or re.search(fc_pattern, key).group(3)
                    )
                    if fc_i != "fc2":
                        merged_moe_ckpts[layer_id][expert_i][key] = torch.cat(
                            [ckpt[key] for ckpt in cur_ckpts], dim=col_dim
                        )
                    else:
                        merged_moe_ckpts[layer_id][expert_i][key] = torch.cat(
                            [ckpt[key] for ckpt in cur_ckpts], dim=row_dim
                        )
                else:
                    raise RuntimeError(f"Unsupported key: {key}")

    # split experts
    moe_str_prefix = ".moe_layer.experts.wrapped_experts."
    split_moe_ckpts = {}
    for _, merged_moe_layer_ckpts in merged_moe_ckpts.items():
        for merged_expert_i in merged_moe_layer_ckpts:
            for key, weight in merged_moe_layer_ckpts[merged_expert_i].items():
                if merge_dim0:
                    weight = weight.reshape(num_local_experts, -1, weight.shape[-1])
                if use_gemm:
                    for i in range(num_local_experts):
                        # get global expert id
                        global_expert_id = merged_expert_i * num_local_experts + i
                        expert_key = key.replace(
                            f"{moe_str_prefix}{merged_expert_i}", f"{moe_str_prefix}{global_expert_id}"
                        )
                        split_moe_ckpts[expert_key] = weight[i].transpose(0, 1).contiguous()
                else:
                    split_moe_ckpts[key] = weight.contiguous()

    return split_moe_ckpts


def merge_pp(states):
    """
    Merge state dicts of pipeline format and shift some layers.

    :return:
        - config: SenseNovaLMConfig
        - states: merged state dict
    """
    # merge pp
    merged_states = []
    for wp_state in tqdm(states):
        assert len(wp_state) == 1, "only pp size 1 is supported currently."
        merged_states.append(wp_state[0])

    return merged_states


def permute(wqkvs, num_shards):
    wqkvs = [wqkvs[i].reshape(3, -1, wqkvs[i].shape[1]) for i in range(num_shards)]
    wq = torch.cat(
        [wqkvs[i][0] for i in range(num_shards)],
        dim=0,
    )
    wk = torch.cat(
        [wqkvs[i][1] for i in range(num_shards)],
        dim=0,
    )
    wv = torch.cat(
        [wqkvs[i][2] for i in range(num_shards)],
        dim=0,
    )
    wqkvs = torch.cat((wq, wk, wv), dim=0)
    return wqkvs


def save_config(tgt):
    model_config = gpc.config.model
    data_config = gpc.config.data
    origin_config = json.load(open("./tools/template.json"))  # pylint: disable=R1732

    origin_config["llm_config"]["vocab_size"] = model_config["vocab_size"]
    origin_config["llm_config"]["hidden_size"] = model_config["hidden_size"]
    origin_config["llm_config"]["num_attention_heads"] = model_config["num_attention_heads"]
    origin_config["llm_config"]["num_hidden_layers"] = model_config["num_layers"]
    origin_config["llm_config"]["num_key_value_heads"] = model_config["num_kv_attention_heads"]
    origin_config["llm_config"]["intermediate_size"] = int(model_config["hidden_size"] * model_config["mlp_ratio"])

    origin_config["llm_config"]["rope_theta"] = model_config["rope_base"]
    origin_config["llm_config"]["rope_scaling"] = {"factor": 2.0, "type": "dynamic"}

    for key in [
        "drop_path_rate",
        "num_attention_heads",
        "num_hidden_layers",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "qkv_bias",
        "proj_bias",
        "qk_normalization",
    ]:
        origin_config["vision_config"][key] = model_config["vit_cfg"][key]

    for key in ["norm_type"]:
        if key in model_config["vit_cfg"]:
            origin_config["vision_config"][key] = model_config["vit_cfg"][key]

    origin_config["select_layer"] = model_config["vision_select_layer"]
    origin_config["image_fold"] = model_config["image_fold"]
    origin_config["ps_version"] = model_config["ps_version"]

    origin_config["vision_config"]["use_moe"] = False

    origin_config["vision_config"]["drop_path_rate"] = model_config["vit_cfg"]["drop_path_rate"]
    if model_config["use_moe"]:
        if model_config["moe_location"] == "llm":
            origin_config["llm_config"]["moe_kwargs"] = model_config["moe_kwargs"]
            origin_config["llm_config"]["moe_layer_kwargs"] = model_config["moe_layer_kwargs"]
        elif model_config["moe_location"] == "vision":
            origin_config["vision_config"]["use_moe"] = True
            origin_config["vision_config"]["moe_cfg"] = model_config["vit_cfg"]["moe_cfg"]
            origin_config["vision_config"]["moe_layer_kwargs"] = model_config["vit_cfg"]["moe_layer_kwargs"]

    origin_config["template"] = data_config["conv_style"]
    origin_config["pad2square"] = data_config["pad2square"]
    origin_config["use_thumbnail"] = data_config["use_thumbnail"]
    origin_config["dynamic_image_size"] = data_config["dynamic_image_size"]
    origin_config["max_dynamic_patch"] = data_config["max_dynamic_patch"]
    origin_config["min_dynamic_patch"] = data_config["min_dynamic_patch"]
    origin_config["vision_config"]["image_size"] = data_config["image_size"]
    origin_config["force_image_size"] = data_config["force_image_size"]
    origin_config["downsample_ratio"] = data_config["down_sample_ratio"]

    def custom_serializer(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)

    # save hf format config
    with open(os.path.join(tgt, "config.json"), "w") as f:
        json.dump(origin_config, f, indent=4)

    # save total model and data config
    with open(os.path.join(tgt, "total_config.json"), "w") as f:
        json.dump(gpc.config, f, indent=4, default=custom_serializer)

    return origin_config


def convert_to_hf(src: str, tgt: str):
    """
    Convert state_dict to hf format.

    1. Save model and data config.
    2. Load and merge state dict.
    3. Convert to huggingface format ckpt.
    4. Load tokneizer and save it with ``tokenizer.save_pretrained``.
    """

    origin_config = save_config(tgt)

    src_states = load_source_ckpt(src)

    # merge pp
    states = merge_pp(src_states)
    del src_states

    num_shards = len(states)
    print(f"Converting model states with wp size:{num_shards} to huggingface format...", flush=True)

    print("Start converting...", flush=True)
    state_dict = {}

    model_config = gpc.config.model
    use_moe = model_config.use_moe
    moe_location = model_config.moe_location
    if use_moe:
        moe_states = load_source_moe_ckpt(src)
        state_dict.update(moe_states)

    # process vision model layers
    for layer_i in tqdm(range(model_config["vit_cfg"]["num_hidden_layers"])):
        if model_config["pure_llm"]:
            break

        state_dict[f"vision_model.encoder.layers.{layer_i}.ls1"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.ls1"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.ls2"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.ls2"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.norm1.weight"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.norm1.weight"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.norm2.weight"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.norm2.weight"
        ]
        if (
            "norm_type" in origin_config["vision_config"]
            and origin_config["vision_config"]["norm_type"] == "layer_norm"
        ):
            state_dict[f"vision_model.encoder.layers.{layer_i}.norm1.bias"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.norm1.bias"
            ]
            state_dict[f"vision_model.encoder.layers.{layer_i}.norm2.bias"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.norm2.bias"
            ]
        if origin_config["vision_config"]["qk_normalization"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.q_norm.weight"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.attn.q_norm.weight"
            ]
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.k_norm.weight"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.attn.k_norm.weight"
            ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.attn.qkv.weight"] = torch.cat(
            [states[i][f"vision_model.encoder.layers.{layer_i}.attn.qkv.weight"] for i in range(num_shards)], dim=0
        )
        if origin_config["vision_config"]["qkv_bias"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.qkv.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.attn.qkv.bias"] for i in range(num_shards)], dim=0
            )
        state_dict[f"vision_model.encoder.layers.{layer_i}.attn.proj.weight"] = torch.cat(
            [states[i][f"vision_model.encoder.layers.{layer_i}.attn.proj.weight"] for i in range(num_shards)], dim=0
        )
        if origin_config["vision_config"]["proj_bias"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.proj.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.attn.proj.bias"] for i in range(num_shards)],
                dim=0,
            )

        if not use_moe or moe_location == "llm":
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc1.weight"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc1.weight"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc1.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc1.bias"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc2.weight"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc2.weight"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc2.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc2.bias"] for i in range(num_shards)],
                dim=0,
            )

    # embedding
    if not model_config["pure_llm"]:
        state_dict["vision_model.embeddings.class_embedding"] = states[0][
            "vision_model.embeddings.class_embedding"
        ].clone()
        state_dict["vision_model.embeddings.position_embedding"] = states[0][
            "vision_model.embeddings.position_embedding"
        ].clone()
        state_dict["vision_model.embeddings.patch_embedding.weight"] = states[0][
            "vision_model.embeddings.patch_embedding.weight"
        ].clone()
        state_dict["vision_model.embeddings.patch_embedding.bias"] = states[0][
            "vision_model.embeddings.patch_embedding.bias"
        ].clone()

    # process language model layers
    qkv_bias = getattr(model_config, "qkv_bias", False) or (not getattr(model_config, "no_bias", True))
    o_bias = getattr(model_config, "o_bias", False) or (not getattr(model_config, "no_bias", True))
    mlp_bias = getattr(model_config, "mlp_bias", False) or (not getattr(model_config, "no_bias", True))
    print(f"model bias setting is: {qkv_bias=} {o_bias=} {mlp_bias=}", flush=True)
    for layer_i in tqdm(range(model_config["num_layers"])):
        state_dict.update(
            {
                f"language_model.model.layers.{layer_i}.attention_norm.weight": states[0][
                    f"language_model.layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"language_model.model.layers.{layer_i}.ffn_norm.weight": states[0][
                    f"language_model.layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
        )
        if f"language_model.layers.{layer_i}.attention.wqkv.weight" in states[0]:
            state_dict[f"language_model.model.layers.{layer_i}.attention.wqkv.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wqkv.weight"] for i in range(num_shards)], dim=0
            )
            if qkv_bias:
                state_dict[f"language_model.model.layers.{layer_i}.attention.wqkv.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wqkv.bias"] for i in range(num_shards)],
                    dim=0,
                )
        else:
            state_dict[f"language_model.model.layers.{layer_i}.attention.wq.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wq.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.attention.wk.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wk.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.attention.wv.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wv.weight"] for i in range(num_shards)], dim=0
            )
            if qkv_bias:
                state_dict[f"language_model.model.layers.{layer_i}.attention.wq.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wq.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.attention.wk.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wk.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.attention.wv.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wv.bias"] for i in range(num_shards)],
                    dim=0,
                )
        state_dict[f"language_model.model.layers.{layer_i}.attention.wo.weight"] = torch.cat(
            [states[i][f"language_model.layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=0
        )
        if o_bias:
            state_dict[f"language_model.model.layers.{layer_i}.attention.wo.bias"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wo.bias"] for i in range(num_shards)], dim=0
            )
        if not use_moe or moe_location == "vision":
            state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )
            if mlp_bias:
                state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w1.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w1.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w2.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w2.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.feed_forward.w3.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w3.bias"] for i in range(num_shards)],
                    dim=0,
                )

    # embedding+head+norm
    state_dict.update(
        {
            "language_model.model.norm.weight": states[0]["language_model.norm.weight"],
            "language_model.model.tok_embeddings.weight": torch.cat(
                [states[i]["language_model.tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
        },
    )
    state_dict["language_model.output.weight"] = torch.cat(
        [states[i]["language_model.output.weight"] for i in range(num_shards)], dim=0
    )

    # mlp
    if not model_config["pure_llm"]:
        state_dict["mlp1.0.weight"] = states[0]["mlp1.0.weight"].clone()
        state_dict["mlp1.0.bias"] = states[0]["mlp1.0.bias"].clone()
        state_dict["mlp1.1.weight"] = torch.cat([states[i]["mlp1.1.weight"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.1.bias"] = torch.cat([states[i]["mlp1.1.bias"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.3.weight"] = torch.cat([states[i]["mlp1.3.weight"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.3.bias"] = torch.cat([states[i]["mlp1.3.bias"] for i in range(num_shards)], dim=0)

    print(f"Start writing safetensors to {tgt} ...", flush=True)

    shards, index = shard_checkpoint(state_dict, weights_name=SAFE_WEIGHTS_NAME)
    for shard_file, shard in shards.items():
        save_file(shard, os.path.join(tgt, shard_file), metadata={"format": "pt"})

    if index is not None:
        save_index_file = os.path.join(tgt, SAFE_WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

    print("Start saving tokenizer ...")
    tokenizer = gpc.tokenizer
    tokenizer.save_pretrained(tgt)


def convert_to_hf_for_qwen(src: str, tgt: str):
    """
    Convert state_dict to hf format.

    1. Save model and data config.
    2. Load and merge state dict.
    3. Convert to huggingface format ckpt.
    4. Load tokneizer and save it with ``tokenizer.save_pretrained``.
    """

    origin_config = save_config(tgt)

    src_states = load_source_ckpt(src)

    # merge pp
    states = merge_pp(src_states)
    del src_states

    num_shards = len(states)
    print(f"Converting model states with wp size:{num_shards} to huggingface format...", flush=True)

    print("Start converting...", flush=True)
    state_dict = {}

    model_config = gpc.config.model
    use_moe = model_config.use_moe
    moe_location = model_config.moe_location
    if use_moe:
        moe_states = load_source_moe_ckpt(src)
        state_dict.update(moe_states)

    # process vision model layers
    for layer_i in tqdm(range(model_config["vit_cfg"]["num_hidden_layers"])):
        if model_config["pure_llm"]:
            break

        state_dict[f"vision_model.encoder.layers.{layer_i}.ls1"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.ls1"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.ls2"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.ls2"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.norm1.weight"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.norm1.weight"
        ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.norm2.weight"] = states[0][
            f"vision_model.encoder.layers.{layer_i}.norm2.weight"
        ]
        if (
            "norm_type" in origin_config["vision_config"]
            and origin_config["vision_config"]["norm_type"] == "layer_norm"
        ):
            state_dict[f"vision_model.encoder.layers.{layer_i}.norm1.bias"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.norm1.bias"
            ]
            state_dict[f"vision_model.encoder.layers.{layer_i}.norm2.bias"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.norm2.bias"
            ]
        if origin_config["vision_config"]["qk_normalization"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.q_norm.weight"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.attn.q_norm.weight"
            ]
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.k_norm.weight"] = states[0][
                f"vision_model.encoder.layers.{layer_i}.attn.k_norm.weight"
            ]
        state_dict[f"vision_model.encoder.layers.{layer_i}.attn.qkv.weight"] = torch.cat(
            [states[i][f"vision_model.encoder.layers.{layer_i}.attn.qkv.weight"] for i in range(num_shards)], dim=0
        )
        if origin_config["vision_config"]["qkv_bias"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.qkv.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.attn.qkv.bias"] for i in range(num_shards)], dim=0
            )
        state_dict[f"vision_model.encoder.layers.{layer_i}.attn.proj.weight"] = torch.cat(
            [states[i][f"vision_model.encoder.layers.{layer_i}.attn.proj.weight"] for i in range(num_shards)], dim=0
        )
        if origin_config["vision_config"]["proj_bias"]:
            state_dict[f"vision_model.encoder.layers.{layer_i}.attn.proj.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.attn.proj.bias"] for i in range(num_shards)],
                dim=0,
            )

        if not use_moe or moe_location == "llm":
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc1.weight"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc1.weight"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc1.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc1.bias"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc2.weight"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc2.weight"] for i in range(num_shards)],
                dim=0,
            )
            state_dict[f"vision_model.encoder.layers.{layer_i}.mlp.fc2.bias"] = torch.cat(
                [states[i][f"vision_model.encoder.layers.{layer_i}.mlp.fc2.bias"] for i in range(num_shards)],
                dim=0,
            )

    # embedding
    if not model_config["pure_llm"]:
        state_dict["vision_model.embeddings.class_embedding"] = states[0][
            "vision_model.embeddings.class_embedding"
        ].clone()
        state_dict["vision_model.embeddings.position_embedding"] = states[0][
            "vision_model.embeddings.position_embedding"
        ].clone()
        state_dict["vision_model.embeddings.patch_embedding.weight"] = states[0][
            "vision_model.embeddings.patch_embedding.weight"
        ].clone()
        state_dict["vision_model.embeddings.patch_embedding.bias"] = states[0][
            "vision_model.embeddings.patch_embedding.bias"
        ].clone()

    # process language model layers
    qkv_bias = getattr(model_config, "qkv_bias", False) or (not getattr(model_config, "no_bias", True))
    o_bias = getattr(model_config, "o_bias", False) or (not getattr(model_config, "no_bias", True))
    mlp_bias = getattr(model_config, "mlp_bias", False) or (not getattr(model_config, "no_bias", True))
    print(f"model bias setting is: {qkv_bias=} {o_bias=} {mlp_bias=}", flush=True)
    for layer_i in tqdm(range(model_config["num_layers"])):
        state_dict.update(
            {
                f"language_model.model.layers.{layer_i}.input_layernorm.weight": states[0][
                    f"language_model.layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"language_model.model.layers.{layer_i}.post_attention_layernorm.weight": states[0][
                    f"language_model.layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
        )
        if f"language_model.layers.{layer_i}.attention.wqkv.weight" in states[0]:
            state_dict[f"language_model.model.layers.{layer_i}.self_attn.wqkv.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wqkv.weight"] for i in range(num_shards)], dim=0
            )
            if qkv_bias:
                state_dict[f"language_model.model.layers.{layer_i}.self_attn.wqkv.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wqkv.bias"] for i in range(num_shards)],
                    dim=0,
                )
        else:
            state_dict[f"language_model.model.layers.{layer_i}.self_attn.q_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wq.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.self_attn.k_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wk.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wv.weight"] for i in range(num_shards)], dim=0
            )
            if qkv_bias:
                state_dict[f"language_model.model.layers.{layer_i}.self_attn.q_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wq.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.self_attn.k_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wk.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.self_attn.v_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.attention.wv.bias"] for i in range(num_shards)],
                    dim=0,
                )
        state_dict[f"language_model.model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
            [states[i][f"language_model.layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=0
        )
        if o_bias:
            state_dict[f"language_model.model.layers.{layer_i}.self_attn.o_proj.bias"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.attention.wo.bias"] for i in range(num_shards)], dim=0
            )
        if not use_moe or moe_location == "vision":
            state_dict[f"language_model.model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"language_model.model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [states[i][f"language_model.layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )
            if mlp_bias:
                state_dict[f"language_model.model.layers.{layer_i}.mlp.gate_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w1.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.mlp.down_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w2.bias"] for i in range(num_shards)],
                    dim=0,
                )
                state_dict[f"language_model.model.layers.{layer_i}.mlp.up_proj.bias"] = torch.cat(
                    [states[i][f"language_model.layers.{layer_i}.feed_forward.w3.bias"] for i in range(num_shards)],
                    dim=0,
                )

    # embedding+head+norm
    state_dict.update(
        {
            "language_model.model.norm.weight": states[0]["language_model.norm.weight"],
            "language_model.model.embed_tokens.weight": torch.cat(
                [states[i]["language_model.tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
        },
    )
    state_dict["language_model.lm_head.weight"] = torch.cat(
        [states[i]["language_model.output.weight"] for i in range(num_shards)], dim=0
    )

    # mlp
    if not model_config["pure_llm"]:
        state_dict["mlp1.0.weight"] = states[0]["mlp1.0.weight"].clone()
        state_dict["mlp1.0.bias"] = states[0]["mlp1.0.bias"].clone()
        state_dict["mlp1.1.weight"] = torch.cat([states[i]["mlp1.1.weight"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.1.bias"] = torch.cat([states[i]["mlp1.1.bias"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.3.weight"] = torch.cat([states[i]["mlp1.3.weight"] for i in range(num_shards)], dim=0)
        state_dict["mlp1.3.bias"] = torch.cat([states[i]["mlp1.3.bias"] for i in range(num_shards)], dim=0)

    print(f"Start writing safetensors to {tgt} ...", flush=True)

    shards, index = shard_checkpoint(state_dict, weights_name=SAFE_WEIGHTS_NAME)
    for shard_file, shard in shards.items():
        save_file(shard, os.path.join(tgt, shard_file), metadata={"format": "pt"})

    if index is not None:
        save_index_file = os.path.join(tgt, SAFE_WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

    print("Start saving tokenizer ...")
    tokenizer = gpc.tokenizer
    tokenizer.save_pretrained(tgt)


def convert_internevo_ckpt_to_hf(src: str, tgt: str):
    assert src is not None, "src dir is needed!"
    start = time.time()

    tgt = tgt.split(":")[-1]
    if not os.path.exists(tgt):
        os.makedirs(tgt)

    if gpc.config.model_type == "QWEN2":
        convert_to_hf_for_qwen(src, tgt)
    else:
        convert_to_hf(src, tgt)

    print(f"Converting model takes {time.time() - start}s totally", flush=True)
