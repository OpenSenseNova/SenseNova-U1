# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from torch import nn

from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.execution_time import execution_time_collecter as etc

with etc.collect_execute_time("import_time"):
    import os

    import torch
    from tqdm import tqdm

    from sensenovalm.accelerator import get_accelerator
    from sensenovalm.core.context import ParallelMode
    from sensenovalm.core.context import global_context as gpc
    from sensenovalm.core.parallel.comm.utils import all_gather_raw
    from sensenovalm.utils.logger import get_logger
    from sensenovalm.utils.parallel import is_using_isp
    from sensenovavl.utils.utils import load_safetensors

# global llm logger
logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


def apply_jitter(x, epsilon, generator):
    """
    Function to apply jitter
    """
    if epsilon == 0:
        return x
    # Create a uniform distribution for jittering
    # uniform = torch.distributions.uniform.Uniform(low=1.0 - epsilon, high=1.0 + epsilon)
    uniform = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) * 2 * epsilon + 1.0 - epsilon
    # Apply jitter by multiplying with a sampled value from the uniform distribution
    jittered_x = x * uniform
    return jittered_x


def _split_state_dict(state_dict):
    """
    This function splits the state_dict since the key is different in loading whole ckpt and seperate ckpt.
    For example:
    1. when loading the whole ckpt using only one path, the key is just like "vision_model.xxx"
    2. when loading the seperate ckpt using three path, the key is just like "encoder.layer.xxx"
    """
    state_dict_vit, state_dict_llm, state_dict_mlp, state_dict_fm_modules = {}, {}, {}, {}

    for name, param in state_dict.items():
        names = name.split(".")
        new_name = ".".join(names[1:])
        if name.startswith("vision_model"):
            state_dict_vit[new_name] = param
        elif name.startswith("language_model"):
            state_dict_llm[new_name] = param
        elif name.startswith("mlp1"):
            state_dict_mlp[new_name] = param
        elif name.startswith("fm_modules"):
            state_dict_fm_modules[new_name] = param

    assert len(state_dict) == len(state_dict_vit) + len(state_dict_llm) + len(state_dict_mlp) + len(state_dict_fm_modules)
    state_dict = None
    return state_dict_vit, state_dict_llm, state_dict_mlp, state_dict_fm_modules


def _resize_vocab_size(state_dict, old_vocab_size, new_vocab_size):
    """
    when the vocab size can not be splitted evenly, the vocab size should be resized.
    """

    def _init_weights(module):
        # NOTE the std is equal to the initializer_range in llm of hugging face, the default is 0.02
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    if gpc.is_rank_for_log():
        print("Resizing Vocab size...", flush=True)

    old_embedding_weight = state_dict["model.tok_embeddings.weight"]
    old_output_weight = state_dict["output.weight"]

    n = min(old_vocab_size, new_vocab_size)

    new_embedding = nn.Embedding(
        new_vocab_size, old_embedding_weight.shape[1], device=get_current_device(), dtype=old_embedding_weight.dtype
    )

    _init_weights(new_embedding)

    new_embedding.weight.data[:n, :] = old_embedding_weight.data[:n, :]

    new_output = nn.Linear(
        old_output_weight.shape[1],
        new_vocab_size,
        device=get_current_device(),
        dtype=old_output_weight.dtype,
    )

    new_output.weight.data[:n, :] = old_output_weight.data[:n, :]
    num_add_token = new_vocab_size - old_vocab_size
    new_output_weight = new_output.weight.data
    output_embeddings_avg = new_output_weight[:-num_add_token].mean(dim=0, keepdim=True)
    new_output_weight[-num_add_token:] = output_embeddings_avg

    state_dict["model.tok_embeddings.weight"] = new_embedding.weight.data
    state_dict["output.weight"] = new_output_weight

    new_embedding = None
    new_output = None


def _load_pretrained_hf_vit(model, model_args, state_dict, has_moe: bool = False):

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    vision_model_state_dict = model.state_dict()

    for name, param in state_dict.items():
        if ".mlp" in name:
            if not has_moe:
                split_size = wp_size if tp_mode == "isp" else tp_size
                local_rank = wp_rank if tp_mode == "isp" else tp_rank
                row_dim = 0 if tp_mode == "isp" else 1
                if "fc1.weight" in name:
                    vision_model_state_dict[name] = torch.chunk(param, split_size, dim=0)[local_rank]
                elif "fc2.weight" in name:
                    vision_model_state_dict[name] = torch.chunk(param, split_size, dim=row_dim)[local_rank]
                elif "fc1.bias" in name:
                    vision_model_state_dict[name] = torch.chunk(param, split_size, dim=0)[local_rank]
                # ignore tp_rank!=0 if fc2.bias
                elif "fc2.bias" in name:
                    if tp_mode == "isp":
                        vision_model_state_dict[name] = torch.chunk(param, split_size, dim=row_dim)[local_rank]
                    else:
                        if tp_rank == 0:
                            vision_model_state_dict[name] = param.clone() if tp_rank == 0 else None
        elif ".qkv.weight" in name:
            if tp_mode == "isp":
                vision_model_state_dict[name] = torch.chunk(param, wp_size, dim=0)[wp_rank]
            else:
                wqkv_split = torch.chunk(param, 3, dim=0)
                q_split = wqkv_split[0]
                k_split = wqkv_split[1]
                v_split = wqkv_split[2]

                num_head = model_args.num_attention_heads
                head_dim = int(param.shape[0] / 3 / num_head)
                div_head = num_head // tp_size
                mod_head = num_head % tp_size
                local_head = div_head + int(tp_rank < mod_head)

                if tp_rank < mod_head:
                    start_head = (div_head + 1) * tp_rank
                else:
                    start_head = (div_head + 1) * mod_head + (tp_rank - mod_head) * div_head
                end_head = start_head + local_head
                q = q_split[start_head * head_dim : end_head * head_dim]
                k = k_split[start_head * head_dim : end_head * head_dim]
                v = v_split[start_head * head_dim : end_head * head_dim]

                vision_model_state_dict[name] = torch.concat([q, k, v], dim=0)

        elif ".qkv.bias" in name:
            if tp_mode == "isp":
                vision_model_state_dict[name] = torch.chunk(param, wp_size, dim=0)[wp_rank]
            else:
                wqkv_split = torch.chunk(param, 3, dim=0)
                q_split = wqkv_split[0]
                k_split = wqkv_split[1]
                v_split = wqkv_split[2]

                num_head = model_args.num_attention_heads
                head_dim = int(param.shape[0] / 3 / num_head)
                div_head = num_head // tp_size
                mod_head = num_head % tp_size
                local_head = div_head + int(tp_rank < mod_head)

                if tp_rank < mod_head:
                    start_head = (div_head + 1) * tp_rank
                else:
                    start_head = (div_head + 1) * mod_head + (tp_rank - mod_head) * div_head
                end_head = start_head + local_head

                q = q_split[start_head * head_dim : end_head * head_dim]
                k = k_split[start_head * head_dim : end_head * head_dim]
                v = v_split[start_head * head_dim : end_head * head_dim]
                vision_model_state_dict[name] = torch.concat([q, k, v], dim=0)
        elif ".proj.weight" in name:
            if tp_mode == "isp":
                vision_model_state_dict[name] = torch.chunk(param, wp_size, dim=0)[wp_rank]
            else:
                num_head = model_args.num_attention_heads
                head_dim = int(param.shape[1] / num_head)
                div_head = num_head // tp_size
                mod_head = num_head % tp_size
                local_head = div_head + int(tp_rank < mod_head)

                if tp_rank < mod_head:
                    start_head = (div_head + 1) * tp_rank
                else:
                    start_head = (div_head + 1) * mod_head + (tp_rank - mod_head) * div_head
                end_head = start_head + local_head
                vision_model_state_dict[name] = param[:, start_head * head_dim : end_head * head_dim]
        elif ".proj.bias" in name:
            if tp_mode == "isp":
                vision_model_state_dict[name] = torch.chunk(param, wp_size, dim=0)[wp_rank]
            else:
                if tp_rank == 0:
                    vision_model_state_dict[name] = param.clone()
        else:
            vision_model_state_dict[name] = param.clone()

    if has_moe:
        moe_states = load_pretrained_vit_moe_online(state_dict=state_dict, model_args=model_args)
        vision_model_state_dict.update(moe_states)

    message = model.load_state_dict(vision_model_state_dict, strict=False)
    del state_dict
    sensenovalm_accelerator.empty_cache()
    return message


def _load_model_checkpoint(folder):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.
    """

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
    dp_size = gpc.get_world_size(ParallelMode.DATA)

    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)

    fns = list(os.listdir(folder))

    _start_with = "model_w" if is_using_isp() else "model_t"

    max_pp, max_wp, max_tp, max_zo = 0, 0, 0, 0
    for fn in fns:
        if fn.startswith(_start_with) and not fn.endswith(".md5"):
            segements = os.path.splitext(fn)[0].split("_")
            if is_using_isp():
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_wp = max(max_wp, int(segements[-2][2:]))
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

    states = torch.load(fp)

    """
    # need convert the gate parameters to float32 (to fit deepspeed style mechanism), it may cause round-off in
    # gate.weight. The conversion will also be done when doing forward. so we can just comment it out. this make
    # the gate parameters to be float16 before forward.
    for key in list(states.keys()):
        if 'moe_layer.gate.wg.weight' in key:
            states[key] = states[key].float()
            print("load: ", states[key].float(),flush=True)
    """

    # try to load expert parameter to separate files if model have moe layer
    # try_load_moe_checkpoint(folder, model, states, tp_rank, pp_rank)

    return states


def _load_hf_checkpoint(folder):
    state_dict = {}
    fns = list(os.listdir(folder))
    for fn in tqdm(fns):
        if fn.endswith(".safetensors"):
            state_dict.update(load_safetensors(os.path.join(folder, fn)))
        elif fn.endswith(".bin"):
            state_dict.update(torch.load(os.path.join(folder, fn)))
        else:
            continue

    return state_dict


def _load_pretrained_internevo_llm(model, model_args, state_dict, has_moe=False):
    language_model_state_dict = model.state_dict()

    for i in range(0, gpc.config.model.num_layers_for_pp):
        language_model_state_dict[f"layers.{i}.attention.wqkv.weight"] = state_dict.pop(
            f"language_model.layers.{i+model.first_layer}.attention.wqkv.weight"
        )

        language_model_state_dict[f"layers.{i}.attention.wo.weight"] = state_dict.pop(
            f"language_model.layers.{i+model.first_layer}.attention.wo.weight"
        )

        if not has_moe:
            language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w1.weight"
            )

            language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w3.weight"
            )

            language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w2.weight"
            )

        language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
            f"language_model.layers.{i+model.first_layer}.attention_norm.weight"
        )
        language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
            f"language_model.layers.{i+model.first_layer}.ffn_norm.weight"
        )

    if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
        language_model_state_dict["tok_embeddings.weight"] = state_dict.pop("language_model.tok_embeddings.weight")

    if (
        os.environ.get("lm_head_no_split", "null").lower() == "true"
        or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
    ):
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = state_dict.pop("language_model.output.weight")
            language_model_state_dict["norm.weight"] = state_dict["language_model.norm.weight"]
    else:
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = state_dict.pop("language_model.output.weight")
            language_model_state_dict["norm.weight"] = state_dict["language_model.norm.weight"]

    if has_moe:
        moe_states = load_pretrained_llm_moe_online(
            model=model, state_dict=state_dict, model_args=model_args, from_hf=False
        )
        language_model_state_dict.update(moe_states)

    message = model.load_state_dict(language_model_state_dict, strict=False)

    return message


def _load_pretrained_hf_llm(model, model_args, model_path=None, state_dict=None, has_moe=False):

    assert gpc.config.model_type in ["QWEN2", "QWEN3_MoE", "QWEN3_MoEMoT"]
    if gpc.config.model_type in ["QWEN2", "QWEN3_MoE", "QWEN3_MoEMoT"]:
        return _load_pretrained_qwen2_llm(model, model_args, model_path, state_dict)
    elif gpc.config.model_type == "SENSENOVALM2_PUBLIC":
        return _load_pretrained_sensenovalm2_llm(model, model_args, model_path, state_dict)
    elif gpc.config.model_type == "SENSENOVALM3_PUBLIC":
        return _load_pretrained_sensenovalm3_llm(model, model_args, model_path, state_dict)
    elif gpc.config.model_type == "DEEPSEEK3_MoE":
        return _load_pretrained_ds3_llm(model, model_args, model_path, state_dict, has_moe)
    else:
        raise ValueError(f"Invalid model type: {gpc.config.model_type}")


def _load_pretrained_sensenovalm2_llm(model, model_args, model_path=None, state_dict=None):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    if model_path is not None:
        if state_dict is None:
            state_dict = {}
        suffix = ".safetensors"
        fns = list(os.listdir(model_path))
        for fn in tqdm(fns):
            if not fn.endswith(suffix):
                continue
            state_dict.update(load_safetensors(os.path.join(model_path, fn)))

        suffix = ".bin"
        fns = list(os.listdir(model_path))
        for fn in tqdm(fns):
            if not fn.endswith(suffix):
                continue
            state_dict.update(torch.load(os.path.join(model_path, fn)))

    # sensenovalm
    if "output.weight" in state_dict:
        old_vocab_size = state_dict["output.weight"].shape[0]
    else:
        raise ValueError("Not Found output weight")

    if old_vocab_size != gpc.config.VOCAB_SIZE:
        print(f"resize vocab size from {old_vocab_size} --> {gpc.config.VOCAB_SIZE}")
        _resize_vocab_size(state_dict, old_vocab_size=old_vocab_size, new_vocab_size=gpc.config.VOCAB_SIZE)

    language_model_state_dict = {}  # model.state_dict()
    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1
    for i in range(0, gpc.config.model.num_layers_for_pp):
        language_model_state_dict[f"layers.{i}.attention.wqkv.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.attention.wqkv.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.attention.wo.weight"),
            split_size,
            dim=row_dim,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w1.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w3.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w2.weight"),
            split_size,
            dim=row_dim,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.attention_norm.weight"
        )
        language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.ffn_norm.weight"
        )

    if model_args.embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0

    if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
        language_model_state_dict["tok_embeddings.weight"] = torch.chunk(
            state_dict.pop("model.tok_embeddings.weight"),
            split_size,
            dim=embed_concat_dim,
        )[local_rank]

    if (
        os.environ.get("lm_head_no_split", "null").lower() == "true"
        or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
    ):
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("output.weight"),
                1,
                dim=0,
            )[0]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    else:
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("output.weight"),
                split_size,
                dim=0,
            )[local_rank]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    message = model.load_state_dict(language_model_state_dict, strict=False)
    del state_dict
    sensenovalm_accelerator.empty_cache()
    return message


def convert_q_k_v_to_wqkv_interleaved(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_outer_heads: int,
    num_key_value_groups: int,
    head_dim: int,
):
    """
    将旧模型的 Q, K, V 权重, 以「每个外层 head 包含 num_key_value_groups 个 Q + 1 K + 1 V」
    的顺序交错拼接, 得到新模型需要的 wqkv.weight.

    参数:
    -------
    q_weight : torch.Tensor
        [num_outer_heads * num_key_value_groups * head_dim, hidden_size]
    k_weight : torch.Tensor
        [num_outer_heads * head_dim, hidden_size]
    v_weight : torch.Tensor
        [num_outer_heads * head_dim, hidden_size]
    num_outer_heads : int
        “外层 head”或“分组”数量.
    num_key_value_groups : int
        对 Q 的分组数 (在同一个 outer head 下, 要多少个子Q).
    head_dim : int
        单个 head 的维度.

    返回:
    -------
    wqkv_weight : torch.Tensor
        [ num_outer_heads * (num_key_value_groups + 2) * head_dim, hidden_size ]
        其中顺序为:
          (Q_{(0,1)} ... Q_{(0,num_key_value_groups)}, K_0, V_0,
           Q_{(1,1)} ... Q_{(1,num_key_value_groups)}, K_1, V_1,
           ...
           Q_{(h-1,1)} ... Q_{(h-1,num_key_value_groups)}, K_{h-1}, V_{h-1})
    """
    hidden_size = q_weight.shape[1]

    # 基本维度检查
    assert q_weight.shape[0] == num_outer_heads * num_key_value_groups * head_dim, (
        f"q_weight.shape[0] = {q_weight.shape[0]}, " f"expected {num_outer_heads * num_key_value_groups * head_dim}"
    )
    assert k_weight.shape[0] == num_outer_heads * head_dim, (
        f"k_weight.shape[0] = {k_weight.shape[0]}, " f"expected {num_outer_heads * head_dim}"
    )
    assert v_weight.shape[0] == num_outer_heads * head_dim, (
        f"v_weight.shape[0] = {v_weight.shape[0]}, " f"expected {num_outer_heads * head_dim}"
    )

    # 目标 wqkv 的大小: [num_outer_heads * (num_key_value_groups + 2) * head_dim, hidden_size]
    out_features = num_outer_heads * (num_key_value_groups + 2) * head_dim
    wqkv_weight = torch.empty((out_features, hidden_size), dtype=q_weight.dtype, device=q_weight.device)

    # 依次往 wqkv_weight 里填充:
    # 对于每个 outer_head i:
    #   写入 (num_key_value_groups 个 Q_i), 然后 1个 K_i, 再 1个 V_i
    for i in range(num_outer_heads):
        # wqkv 里该 outer_head 的起始位置
        base = i * ((num_key_value_groups + 2) * head_dim)

        # Q_i 在 q_weight 中的起始位置 (Q 的切片大小 = num_key_value_groups * head_dim)
        q_start = i * num_key_value_groups * head_dim

        # 填 Q_i (共 num_key_value_groups * head_dim 行)
        wqkv_weight[base : base + num_key_value_groups * head_dim, :] = q_weight[
            q_start : q_start + num_key_value_groups * head_dim, :
        ]

        # 填 K_i (共 head_dim 行)
        k_start = i * head_dim
        wqkv_weight[
            base + num_key_value_groups * head_dim : base + (num_key_value_groups + 1) * head_dim, :
        ] = k_weight[k_start : k_start + head_dim, :]

        # 填 V_i (共 head_dim 行)
        v_start = i * head_dim
        wqkv_weight[
            base + (num_key_value_groups + 1) * head_dim : base + (num_key_value_groups + 2) * head_dim, :
        ] = v_weight[v_start : v_start + head_dim, :]

    return wqkv_weight


def _load_pretrained_sensenovalm3_llm(model, model_args, model_path=None, state_dict=None):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    if model_path is not None:
        if state_dict is None:
            state_dict = {}
        suffix = ".safetensors"
        fns = list(os.listdir(model_path))
        for fn in tqdm(fns):
            if not fn.endswith(suffix):
                continue
            state_dict.update(load_safetensors(os.path.join(model_path, fn)))

        suffix = ".bin"
        fns = list(os.listdir(model_path))
        for fn in tqdm(fns):
            if not fn.endswith(suffix):
                continue
            state_dict.update(torch.load(os.path.join(model_path, fn)))

    # sensenovalm
    if "lm_head.weight" in state_dict:
        old_vocab_size = state_dict["lm_head.weight"].shape[0]
    else:
        raise ValueError("Not Found output weight")

    if old_vocab_size != gpc.config.VOCAB_SIZE:
        print(f"resize vocab size from {old_vocab_size} --> {gpc.config.VOCAB_SIZE}")
        _resize_vocab_size(  # pylint: disable=E1123
            state_dict,
            old_vocab_size=old_vocab_size,
            new_vocab_size=gpc.config.VOCAB_SIZE,
            input_key="model.embed_tokens.weight",
            output_key="lm_head.weight",
        )

    language_model_state_dict = {}  # model.state_dict()
    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1
    for i in range(0, gpc.config.model.num_layers_for_pp):
        qkv_weights = convert_q_k_v_to_wqkv_interleaved(
            q_weight=state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
            k_weight=state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
            v_weight=state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
            num_outer_heads=gpc.config.model.num_kv_attention_heads,
            num_key_value_groups=gpc.config.model.num_attention_heads // gpc.config.model.num_kv_attention_heads,
            head_dim=gpc.config.model.hidden_size // gpc.config.model.num_attention_heads,
        )
        language_model_state_dict[f"layers.{i}.attention.wqkv.weight"] = torch.chunk(
            qkv_weights,
            split_size,
            dim=0,
        )[local_rank]


        language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
            split_size,
            dim=row_dim,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
            split_size,
            dim=row_dim,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.input_layernorm.weight"
        )
        language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
        )

    if model_args.embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0

    if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
        language_model_state_dict["tok_embeddings.weight"] = torch.chunk(
            state_dict.pop("model.embed_tokens.weight"),
            split_size,
            dim=embed_concat_dim,
        )[local_rank]

    if (
        os.environ.get("lm_head_no_split", "null").lower() == "true"
        or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
    ):
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("lm_head.weight"),
                1,
                dim=0,
            )[0]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    else:
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("lm_head.weight"),
                split_size,
                dim=0,
            )[local_rank]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    message = model.load_state_dict(language_model_state_dict, strict=False)
    del state_dict
    sensenovalm_accelerator.empty_cache()
    return message


def _load_qwen3_moe_experts_from_hf(state_dict, model, model_args, load_mot_gen: bool):
    """Pack per-expert HF weights into the internal grouped-MoE state dict.

    Source layout (e.g. ``SenseNova-U1-A3B-SFT``; experts live in the ``moemodel-*-of-NNNNN.safetensors``
    shards, prefix already stripped to ``model.``)::

        model.layers.{l}.mlp.experts.{e}.gate_proj.weight     # understanding branch, ``num_experts`` experts
        model.layers.{l}.mlp.experts.{e}.up_proj.weight
        model.layers.{l}.mlp.experts.{e}.down_proj.weight
        model.layers.{l}.mlp.gate.weight                      # router
        model.layers.{l}.mlp_mot_gen.experts.{e}.* / .gate.weight   # generation branch, ``gen_num_experts`` experts

    Target layout (``layers`` already pp-local)::

        layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.{k}.{w1,w3,w2}.weight   # w1<-gate, w3<-up, w2<-down
        layers.{i}.feed_forward.moe_layer.gate.wg.weight
        layers.{i}.feed_forward_mot_gen.moe_layer.*                                       # when ``load_mot_gen``

    This mirrors the expert packing in :func:`load_pretrained_llm_moe_online` (transpose for grouped
    gemm, expert-parallel sharding by ``ep_rank``, expert-weight-parallel sharding by ``ewp_rank`` under
    isp, ``merge_dim0``/``stack``/per-expert layout), but consumes the *real* per-expert weights instead
    of up-cycling a dense FFN.  Note: the caller's ``state_dict`` already holds every safetensors tensor
    in CPU memory, so each rank reads the whole (large) MoE checkpoint -- weights are moved to the
    current device one expert at a time and the ones this rank does not own are dropped immediately.
    """
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)
    tp_mode = gpc.config.parallel.tensor["mode"]
    if tp_mode == "isp":
        ewp_size = gpc.get_world_size(ParallelMode.EXPERT_WEIGHT)
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
    else:
        ewp_size, ewp_rank = 1, 0

    row_dim = 0 if tp_mode == "isp" else 1
    col_dim = 0

    moe_kwargs = model_args.moe_kwargs
    moe_type = moe_kwargs.moe_type
    use_gemm = True
    if moe_type == "GShard" and model_args.moe_layer_kwargs.get("use_grouped_mlp", True):
        use_gemm = False
    merge_dim0 = (use_gemm and tp_mode == "isp") or moe_type == "MegaBlock-D"

    jitter_scale = moe_kwargs.get("param_jitter_epsilon", 0.0)
    generator = torch.Generator()
    generator.manual_seed(666)  # keep the jitter consistent across ranks

    num_experts_und = moe_kwargs.num_experts
    num_experts_gen = moe_kwargs.get("gen_num_experts", num_experts_und)
    branches = [("mlp", "feed_forward", num_experts_und)]
    if load_mot_gen:
        branches.append(("mlp_mot_gen", "feed_forward_mot_gen", num_experts_gen))

    moe_state_dict = {}
    for hf_branch, model_branch, num_experts in branches:
        assert num_experts % ep_size == 0, f"{model_branch}: {num_experts=} not divisible by {ep_size=}"
        num_local_experts = num_experts // ep_size
        wrapped_prefix = f"{model_branch}.moe_layer.experts.wrapped_experts"
        for i in range(0, gpc.config.model.num_layers_for_pp):
            layer_idx = i + model.first_layer
            local_w1, local_w3, local_w2 = [], [], []  # w1<-gate_proj, w3<-up_proj, w2<-down_proj
            for e in range(num_experts):
                gate_w = state_dict.pop(f"model.layers.{layer_idx}.{hf_branch}.experts.{e}.gate_proj.weight")
                up_w = state_dict.pop(f"model.layers.{layer_idx}.{hf_branch}.experts.{e}.up_proj.weight")
                down_w = state_dict.pop(f"model.layers.{layer_idx}.{hf_branch}.experts.{e}.down_proj.weight")
                if not (ep_rank * num_local_experts <= e < (ep_rank + 1) * num_local_experts):
                    del gate_w, up_w, down_w
                    continue
                gate_w = gate_w.to(get_current_device())
                up_w = up_w.to(get_current_device())
                down_w = down_w.to(get_current_device())
                if use_gemm:
                    # internal grouped-gemm experts are stored input-major: w1/w3 [hidden, moe_int], w2 [moe_int, hidden]
                    gate_w = gate_w.T
                    up_w = up_w.T
                    down_w = down_w.T
                local_w1.append(gate_w)
                local_w3.append(up_w)
                local_w2.append(down_w)

            if merge_dim0:
                # grouped-gemm under isp: all `num_local_experts` of this ep_rank are concatenated along dim 0
                # (expert-major: each expert occupies a contiguous block), then expert-weight-parallel shards
                # that same dim 0 -- i.e. ewp partitions *which experts* land on each ewp rank, NOT each expert's
                # hidden dim. (This matches how try_save_moe_checkpoint stored the EVO shards.)
                w1 = torch.cat(local_w1, dim=0)  # [num_local_experts * hidden, moe_int]
                w3 = torch.cat(local_w3, dim=0)  # [num_local_experts * hidden, moe_int]
                w2 = torch.cat(local_w2, dim=0)  # [num_local_experts * moe_int, hidden]
                if tp_mode == "isp" and ewp_size > 1:
                    w1 = w1.chunk(ewp_size, dim=0)[ewp_rank].contiguous()
                    w3 = w3.chunk(ewp_size, dim=0)[ewp_rank].contiguous()
                    w2 = w2.chunk(ewp_size, dim=0)[ewp_rank].contiguous()
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w1.weight"] = apply_jitter(w1, jitter_scale, generator)
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w3.weight"] = apply_jitter(w3, jitter_scale, generator)
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w2.weight"] = apply_jitter(w2, jitter_scale, generator)
            elif use_gemm:
                # non-isp grouped gemm: stack along a new expert dim; no expert-weight-parallel here
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w1.weight"] = apply_jitter(torch.stack(local_w1, dim=0), jitter_scale, generator)
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w3.weight"] = apply_jitter(torch.stack(local_w3, dim=0), jitter_scale, generator)
                moe_state_dict[f"layers.{i}.{wrapped_prefix}.0.w2.weight"] = apply_jitter(torch.stack(local_w2, dim=0), jitter_scale, generator)
            else:
                # one nn.Module per local expert: standard column/row tensor (weight) parallel split
                for k in range(num_local_experts):
                    w1k, w3k, w2k = local_w1[k], local_w3[k], local_w2[k]
                    if tp_mode == "isp" and ewp_size > 1:
                        w1k = w1k.chunk(ewp_size, dim=col_dim)[ewp_rank].contiguous()
                        w3k = w3k.chunk(ewp_size, dim=col_dim)[ewp_rank].contiguous()
                        w2k = w2k.chunk(ewp_size, dim=row_dim)[ewp_rank].contiguous()
                    moe_state_dict[f"layers.{i}.{wrapped_prefix}.{k}.w1.weight"] = apply_jitter(w1k, jitter_scale, generator)
                    moe_state_dict[f"layers.{i}.{wrapped_prefix}.{k}.w3.weight"] = apply_jitter(w3k, jitter_scale, generator)
                    moe_state_dict[f"layers.{i}.{wrapped_prefix}.{k}.w2.weight"] = apply_jitter(w2k, jitter_scale, generator)

            gate_key = f"model.layers.{layer_idx}.{hf_branch}.gate.weight"
            if gate_key in state_dict:
                moe_state_dict[f"layers.{i}.{model_branch}.moe_layer.gate.wg.weight"] = state_dict.pop(gate_key).to(get_current_device())

    return moe_state_dict


def _load_pretrained_qwen2_llm(model, model_args, model_path=None, state_dict=None):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]
    
    if model_path is not None:
        if state_dict is None:
            state_dict = {}
        suffix = ".safetensors"
        fns = list(os.listdir(model_path))
        for fn in tqdm(fns):
            if not fn.endswith(suffix):
                continue
            state_dict.update(load_safetensors(os.path.join(model_path, fn)))

    # qwen2 model
    if "lm_head.weight" in state_dict:
        old_vocab_size = state_dict["lm_head.weight"].shape[0]
    elif "model.embed_tokens.weight" in state_dict:
        # tie weights
        old_vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
    else:
        raise ValueError("Not Found output weight")

    if old_vocab_size != gpc.config.VOCAB_SIZE:
        _resize_vocab_size(state_dict, old_vocab_size=old_vocab_size, new_vocab_size=gpc.config.VOCAB_SIZE)

    language_model_state_dict = {}  # model.state_dict()

    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1

    model_config = gpc.config.model

    # MoE language model (e.g. ``SenseNova-U1-A3B-SFT``): the dense FFN weights are replaced by
    # per-expert weights (``model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight``) plus a
    # router (``model.layers.{i}.mlp.gate.weight``). The dense paths below skip the FFN load and the
    # experts are packed into the internal grouped-MoE layout by ``_load_qwen3_moe_experts_from_hf``.
    num_llm_experts = model_config.moe_kwargs.get("num_experts", 1)
    has_llm_moe = num_llm_experts > 1

    qkv_bias = getattr(model_config, "qkv_bias", False) or (not getattr(model_config, "no_bias", True))
    o_bias = getattr(model_config, "o_bias", False) or (not getattr(model_config, "no_bias", True))
    mlp_bias = getattr(model_config, "mlp_bias", False) or (not getattr(model_config, "no_bias", True))
    print(f"model bias setting is: {qkv_bias=} {o_bias=} {mlp_bias=}", flush=True)

    mot_model = getattr(model_args, 'mot_model', False)
    mot_random_init = getattr(model_args, 'mot_random_init', True)
    if mot_model:
        if 'model.layers.0.self_attn.q_proj_mot_gen.weight' in state_dict:
            logger.info('Initializing MoT VLM from one pre-trained MoT VLM')
            language_model_state_dict["norm_mot_gen.weight"] = state_dict.pop("model.norm_mot_gen.weight")
            for i in range(0, gpc.config.model.num_layers_for_pp):
                # norm
                language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
                )
                language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.input_layernorm.weight"
                )

                language_model_state_dict[f"layers.{i}.ffn_norm_mot_gen.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.post_attention_layernorm_mot_gen.weight"
                )
                language_model_state_dict[f"layers.{i}.attention_norm_mot_gen.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.input_layernorm_mot_gen.weight"
                )
                # mlp (dense FFN only; MoE experts are loaded by _load_qwen3_moe_experts_from_hf below)
                if not has_llm_moe:
                    language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
                        split_size,
                        dim=0,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]

                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w1.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp_mot_gen.gate_proj.weight"),
                        split_size,
                        dim=0,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w2.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp_mot_gen.down_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w3.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp_mot_gen.up_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                # attention
                language_model_state_dict[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]

                language_model_state_dict[f"layers.{i}.attention.wq_mot_gen.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj_mot_gen.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wk_mot_gen.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj_mot_gen.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wv_mot_gen.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj_mot_gen.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wo_mot_gen.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj_mot_gen.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]

                # qk_norm
                if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                    norm_weight = state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_norm.weight") 
                    language_model_state_dict[f"layers.{i}.attention.q_norm.weight"] = norm_weight

                    language_model_state_dict[f"layers.{i}.attention.q_norm_hw.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.q_norm_hw.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm_hw.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm_hw.weight"
                    )

                    language_model_state_dict[f"layers.{i}.attention.q_norm_mot_gen.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.q_norm_mot_gen.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.q_norm_hw_mot_gen.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.q_norm_hw_mot_gen.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm_mot_gen.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm_mot_gen.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm_hw_mot_gen.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm_hw_mot_gen.weight"
                    )
        else:
            if mot_random_init:
                logger.info('Initializing MoT VLM from Dense VLM, the generation branch will be randomly initialized')
                for i in range(0, gpc.config.model.num_layers_for_pp):
                    # norm
                    language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.input_layernorm.weight"
                    )
                    # mlp
                    language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
                        split_size,
                        dim=0,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    # attention
                    language_model_state_dict[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]

                    # qk_norm
                    if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                        norm_weight = state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_norm.weight") 
                        language_model_state_dict[f"layers.{i}.attention.q_norm.weight"] = norm_weight

                        language_model_state_dict[f"layers.{i}.attention.q_norm_hw.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.q_norm_hw.weight"
                        )
                        language_model_state_dict[f"layers.{i}.attention.k_norm.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.k_norm.weight"
                        )
                        language_model_state_dict[f"layers.{i}.attention.k_norm_hw.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.k_norm_hw.weight"
                        )
                
                for j in range(0, gpc.config.model.num_layers_for_pp):
                    if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                        # qk norm
                        for name in ["q", "k"]:
                            language_model_state_dict[f"layers.{j}.attention.{name}_norm_mot_gen.weight"] = \
                                torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)

                            for axis in ["hw"]:
                                language_model_state_dict[f"layers.{j}.attention.{name}_norm_{axis}_mot_gen.weight"] = \
                                    torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)
            else:
                logger.info('Initializing the generation branch from the pre-trained Dense VLM')

                for i in range(0, gpc.config.model.num_layers_for_pp):
                    # norm
                    language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.input_layernorm.weight"
                    )

                    language_model_state_dict[f"layers.{i}.ffn_norm_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.ffn_norm.weight"]
                    language_model_state_dict[f"layers.{i}.attention_norm_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.attention_norm.weight"]
                    # mlp
                    language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
                        split_size,
                        dim=0,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]

                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w1.weight"] = \
                        language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"]
                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w2.weight"] = \
                        language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"]
                    language_model_state_dict[f"layers.{i}.feed_forward_mot_gen.w3.weight"] = \
                        language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"]
                    # attention
                    language_model_state_dict[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]
                    language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                        state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
                        split_size,
                        dim=row_dim,
                    )[local_rank]

                    language_model_state_dict[f"layers.{i}.attention.wq_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.attention.wq.weight"]
                    language_model_state_dict[f"layers.{i}.attention.wk_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.attention.wk.weight"] 
                    language_model_state_dict[f"layers.{i}.attention.wv_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.attention.wv.weight"]
                    language_model_state_dict[f"layers.{i}.attention.wo_mot_gen.weight"] = \
                        language_model_state_dict[f"layers.{i}.attention.wo.weight"]
                    # qk_norm
                    if gpc.config.model_type in ["QWEN3_MoE", "QWEN3_MoEMoT"]:
                        language_model_state_dict[f"layers.{i}.attention.q_norm.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.q_norm.weight"
                        )
                        language_model_state_dict[f"layers.{i}.attention.q_norm_hw.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.q_norm_hw.weight"
                        )
                        language_model_state_dict[f"layers.{i}.attention.k_norm.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.k_norm.weight"
                        )
                        language_model_state_dict[f"layers.{i}.attention.k_norm_hw.weight"] = state_dict.pop(
                            f"model.layers.{i+model.first_layer}.self_attn.k_norm_hw.weight"
                        )

                        language_model_state_dict[f"layers.{i}.attention.q_norm_mot_gen.weight"] = \
                            language_model_state_dict[f"layers.{i}.attention.q_norm.weight"]
                        language_model_state_dict[f"layers.{i}.attention.q_norm_hw_mot_gen.weight"] = \
                            language_model_state_dict[f"layers.{i}.attention.q_norm_hw.weight"]
                        language_model_state_dict[f"layers.{i}.attention.k_norm_mot_gen.weight"] = \
                            language_model_state_dict[f"layers.{i}.attention.k_norm.weight"]
                        language_model_state_dict[f"layers.{i}.attention.k_norm_hw_mot_gen.weight"] = \
                            language_model_state_dict[f"layers.{i}.attention.k_norm_hw.weight"]
    else:
        # for i in range(0, gpc.config.model.num_layers_for_pp):
        # NOTE: postlayer ---- #
        if 'model.layers.0.self_attn.q_proj_hw.weight' not in state_dict and False:
            raise NotImplementedError
            logger.info('Initializing dense VLM from one pre-trained dense LLM')
            for i in range(0, model_config.num_layers - model_config.extra_num_layers):
                # norm
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.ffn_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
                )
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.input_layernorm.weight"
                )
                # mlp
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.feed_forward.w1.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
                    split_size,
                    dim=0,
                )[local_rank]
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.feed_forward.w2.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.feed_forward.w3.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                # attention
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wq.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wk.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                    
                # NOTE: rope model
                # TODO: support interleaved rope embedding, now only support half and half
                def initialize_equal_weight_for_HW(model_weight, head_dim):
                    model_weight_t = model_weight.permute(1, 0)
                    num_head = model_weight_t.shape[1] // head_dim
                    model_reshape = model_weight_t.reshape(model_weight_t.shape[0], num_head, head_dim)
                    model_w1, _, model_w3, _ = model_reshape.chunk(4, dim=-1)
                    model_weight_hw = torch.cat([model_w1, model_w3], dim=-1).repeat(1, 1, 2).reshape(model_weight_t.shape[0], -1)
                    return model_weight_hw.permute(1, 0)
                        
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wq_hw.weight"] = \
                    initialize_equal_weight_for_HW(language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wq.weight"], model_config.head_dim)

                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wk_hw.weight"] = \
                    torch.zeros_like(language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wk.weight"])

                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wv.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.wo.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]

                # qk_norm
                if gpc.config.model_type == "QWEN3_MoE":

                    # NOTE: rope model
                    for name in ["q", "k"]:
                        norm_weight = state_dict.pop(
                            f"model.layers.{i + model.first_layer}.self_attn.{name}_norm.weight"
                        )
                        language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.{name}_norm.weight"] = norm_weight

                        half_shape = norm_weight.shape[0]
                        half_ones = torch.ones(half_shape, device=norm_weight.device, dtype=norm_weight.dtype)

                        for axis in ["h", "w"]:
                            language_model_state_dict[f"layers.{i+model_config.extra_num_layers}.attention.{name}_norm_{axis}.weight"] = half_ones.clone()
            
            # NOTE: postlayer ---- #
            for j in range(model_config.extra_num_layers):
                if gpc.config.model_type == "QWEN3_MoE":
                    # qk norm
                    for name in ["q", "k"]:
                        language_model_state_dict[f"layers.{j}.attention.{name}_norm.weight"] = \
                            torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)

                        for axis in ["hw"]:
                            language_model_state_dict[f"layers.{j}.attention.{name}_norm_{axis}.weight"] = \
                                torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)
        else:
            logger.info('Initializing dense VLM from one pre-trained dense VLM')

            # post-buffer layers
            extra_num_layers_post = getattr(model_config, 'extra_num_layers_post', 0)
            if extra_num_layers_post > 0 and f'model.layers.{gpc.config.model.num_layers_for_pp-1}.self_attn.q_norm_hw.weight' not in state_dict:
                logger.info('Initializing post-buffer layers from scratch')
                existing_num_layers = gpc.config.model.num_layers_for_pp - extra_num_layers_post

                for j in range(existing_num_layers, gpc.config.model.num_layers_for_pp):
                    if gpc.config.model_type == "QWEN3_MoE":
                        # qk norm
                        for name in ["q", "k"]:
                            norm_weight = state_dict[f"model.layers.{model.first_layer}.self_attn.{name}_norm.weight"]
                            language_model_state_dict[f"layers.{j}.attention.{name}_norm.weight"] = \
                                torch.ones(norm_weight.shape[0], device=norm_weight.device, dtype=norm_weight.dtype)

                            for axis in ["hw"]:
                                language_model_state_dict[f"layers.{j}.attention.{name}_norm_{axis}.weight"] = \
                                    torch.ones(norm_weight.shape[0] // 2, device=norm_weight.device, dtype=norm_weight.dtype)
            else:
                existing_num_layers = gpc.config.model.num_layers_for_pp


            for i in range(0, existing_num_layers):
                # norm
                language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.post_attention_layernorm.weight"
                )
                language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
                    f"model.layers.{i+model.first_layer}.input_layernorm.weight"
                )
                # mlp
                language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.gate_proj.weight"),
                    split_size,
                    dim=0,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.down_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.mlp.up_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                # attention
                language_model_state_dict[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.q_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.k_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.v_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]
                language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                    state_dict.pop(f"model.layers.{i+model.first_layer}.self_attn.o_proj.weight"),
                    split_size,
                    dim=row_dim,
                )[local_rank]

                # qk_norm
                if gpc.config.model_type == "QWEN3_MoE":
                    language_model_state_dict[f"layers.{i}.attention.q_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.q_norm.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.q_norm_hw.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.q_norm_hw.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm.weight"
                    )
                    language_model_state_dict[f"layers.{i}.attention.k_norm_hw.weight"] = state_dict.pop(
                        f"model.layers.{i+model.first_layer}.self_attn.k_norm_hw.weight"
                    )

    if model_args.embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0

    if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
        language_model_state_dict["tok_embeddings.weight"] = torch.chunk(
            state_dict.pop("model.embed_tokens.weight"),
            split_size,
            dim=embed_concat_dim,
        )[local_rank]

    if "lm_head.weight" in state_dict:
        if (
            os.environ.get("lm_head_no_split", "null").lower() == "true"
            or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
        ):
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                language_model_state_dict["output.weight"] = torch.chunk(
                    state_dict.pop("lm_head.weight"),
                    1,
                    dim=0,
                )[0]
        else:
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                language_model_state_dict["output.weight"] = torch.chunk(
                    state_dict.pop("lm_head.weight"),
                    split_size,
                    dim=0,
                )[local_rank]

    if "output.weight" not in language_model_state_dict:
        # tie word embeddings
        language_model_state_dict["output.weight"] = language_model_state_dict["tok_embeddings.weight"]

    language_model_state_dict["norm.weight"] = state_dict.pop("model.norm.weight")

    if has_llm_moe:
        # Pack the per-expert HF weights (``moemodel-*`` shards) into the internal grouped-MoE layout.
        # Only load the MoT generation experts when they are actually present in the checkpoint
        # (continued-MoT-training case); otherwise the generation branch is initialised elsewhere.
        load_mot_gen_experts = mot_model and (
            f"model.layers.{model.first_layer}.mlp_mot_gen.experts.0.gate_proj.weight" in state_dict
        )
        moe_states = _load_qwen3_moe_experts_from_hf(
            state_dict, model, model_args, load_mot_gen=load_mot_gen_experts
        )
        language_model_state_dict.update(moe_states)

    message = model.load_state_dict(language_model_state_dict, strict=False)

    return message


def _load_pretrained_ds3_llm(model, model_args, model_path=None, state_dict=None, has_moe=False):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    if model_path is not None:
        state_dict = _load_hf_checkpoint(model_path)

    old_vocab_size = state_dict["output.weight"].shape[0]

    if old_vocab_size != gpc.config.VOCAB_SIZE:
        _resize_vocab_size(state_dict, old_vocab_size=old_vocab_size, new_vocab_size=gpc.config.VOCAB_SIZE)

    language_model_state_dict = model.state_dict()

    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1
    for i in range(0, gpc.config.model.num_layers_for_pp):
        language_model_state_dict[f"layers.{i}.attention.wqkv.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.attention.wqkv.weight"),
            split_size,
            dim=0,
        )[local_rank]
        language_model_state_dict[f"layers.{i}.attention.wo.weight"] = torch.chunk(
            state_dict.pop(f"model.layers.{i+model.first_layer}.attention.wo.weight"),
            split_size,
            dim=row_dim,
        )[local_rank]
        if not has_moe:
            language_model_state_dict[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w1.weight"),
                split_size,
                dim=0,
            )[local_rank]
            language_model_state_dict[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w3.weight"),
                split_size,
                dim=0,
            )[local_rank]
            language_model_state_dict[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w2.weight"),
                split_size,
                dim=row_dim,
            )[local_rank]
        language_model_state_dict[f"layers.{i}.attention_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.attention_norm.weight"
        )
        language_model_state_dict[f"layers.{i}.ffn_norm.weight"] = state_dict.pop(
            f"model.layers.{i+model.first_layer}.ffn_norm.weight"
        )

    if model_args.embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0

    if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
        language_model_state_dict["tok_embeddings.weight"] = torch.chunk(
            state_dict.pop("model.tok_embeddings.weight"),
            split_size,
            dim=embed_concat_dim,
        )[local_rank]

    if (
        os.environ.get("lm_head_no_split", "null").lower() == "true"
        or os.environ.get("lm_head_seq_parallel", "null").lower() == "true"
    ):
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("output.weight"),
                1,
                dim=0,
            )[0]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]
    else:
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            language_model_state_dict["output.weight"] = torch.chunk(
                state_dict.pop("output.weight"),
                split_size,
                dim=0,
            )[local_rank]
            language_model_state_dict["norm.weight"] = state_dict["model.norm.weight"]

    if has_moe:
        moe_states = load_pretrained_llm_moe_online(
            model=model, state_dict=state_dict, model_args=model_args, from_hf=True
        )
        language_model_state_dict.update(moe_states)

    message = model.load_state_dict(language_model_state_dict, strict=False)

    del state_dict
    sensenovalm_accelerator.empty_cache()
    return message


def load_pretrained_llm(model, model_args, model_path):
    is_hf_ckpt = True
    suffix = ".pt"
    fns = list(os.listdir(model_path))
    for fn in tqdm(fns):
        if fn.endswith(suffix):
            is_hf_ckpt = False
            break

    use_moe = gpc.config.model.use_moe
    moe_location = gpc.config.model.moe_location
    has_llm_moe = use_moe and moe_location == "llm"

    if not is_hf_ckpt:
        state_dict = _load_model_checkpoint(model_path)
        message = _load_pretrained_internevo_llm(
            model=model, model_args=model_args, state_dict=state_dict, has_moe=has_llm_moe
        )
    else:
        state_dict = _load_hf_checkpoint(model_path)
        message = _load_pretrained_hf_llm(
            model=model, model_args=model_args, state_dict=state_dict, has_moe=has_llm_moe
        )

    del state_dict
    # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
    sensenovalm_accelerator.empty_cache()

    return message


def load_pretrained_mlp(model, model_path=None, state_dict=None):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    if model_path is not None:
        state_dict = torch.load(model_path)

    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1

    mlp_state_dict = model.state_dict()
    mlp_state_dict["0.weight"] = state_dict.pop("0.weight")
    mlp_state_dict["0.bias"] = state_dict.pop("0.bias")

    mlp_state_dict["1.weight"] = torch.chunk(
        state_dict.pop("1.weight"),
        split_size,
        dim=0,
    )[local_rank]
    mlp_state_dict["1.bias"] = torch.chunk(
        state_dict.pop("1.bias"),
        split_size,
        dim=0,
    )[local_rank]

    mlp_state_dict["3.weight"] = torch.chunk(
        state_dict.pop("3.weight"),
        split_size,
        dim=row_dim,
    )[local_rank]
    if tp_mode == "isp":
        mlp_state_dict["3.bias"] = torch.chunk(
            state_dict.pop("3.bias"),
            split_size,
            dim=row_dim,
        )[local_rank]
    else:
        if tp_rank == 0:
            mlp_state_dict["3.bias"] = state_dict["3.bias"].clone()

    message = model.load_state_dict(mlp_state_dict, strict=False)

    sensenovalm_accelerator.empty_cache()
    return message

def load_pretrained_fm_modules(model, model_path=None, state_dict=None):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]

    if model_path is not None:
        state_dict = torch.load(model_path)

    split_size = wp_size if tp_mode == "isp" else tp_size
    local_rank = wp_rank if tp_mode == "isp" else tp_rank
    row_dim = 0 if tp_mode == "isp" else 1

    message = model.load_state_dict(state_dict, strict=False)

    sensenovalm_accelerator.empty_cache()
    return message


def load_pretrained_vit(model, model_args, model_path):
    use_moe = gpc.config.model.use_moe
    moe_location = gpc.config.model.moe_location
    has_vit_moe = use_moe and moe_location == "vision"
    state_dict = _load_hf_checkpoint(model_path)

    message = _load_pretrained_hf_vit(model, model_args, state_dict, has_moe=has_vit_moe)

    del state_dict
    sensenovalm_accelerator.empty_cache()

    return message


def load_pretrained_vit_moe_online(state_dict, model_args):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)

    row_dim = 0 if tp_mode == "isp" else 1
    col_dim = 0

    if tp_mode == "isp":
        ewp_size = gpc.get_world_size(ParallelMode.EXPERT_WEIGHT)
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
    else:
        if gpc.config.parallel.expert.no_tp:
            etp_size = 1
            etp_rank = 0
        else:
            etp_size = tp_size
            etp_rank = tp_rank

    jitter_scale = model_args.moe_cfg.get("param_jitter_epsilon", 0.0)
    generator = torch.Generator()
    generator.manual_seed(666)  # 保证进程间加的jitter是一致的

    merge_dim0 = False
    use_gemm = True
    moe_type = model_args.moe_cfg.moe_type
    num_experts = model_args.moe_cfg.num_experts
    num_local_experts = num_experts // ep_size
    num_shared_experts = model_args.moe_cfg.get("num_shared_experts", 0)
    if moe_type == "GShard" and model_args.moe_layer_kwargs.get("use_grouped_mlp", True):
        use_gemm = False

    if (use_gemm and tp_mode == "isp") or moe_type == "MegaBlock-D":
        merge_dim0 = True

    if use_gemm and gpc.is_rank_for_log():
        logger.warning("use gemm is True, the bias of fc1 and fc2 will be ignored, please note this!")

    split_size = model_args.moe_cfg.get("split_size", model_args.moe_cfg.top_k)
    upcycle_size = num_experts // split_size

    final_moe_state = {}
    vit_num_layers = gpc.config.model.vit_cfg.num_hidden_layers
    for i in range(0, vit_num_layers):
        fc1_weight: torch.Tensor = state_dict.pop(f"encoder.layers.{i}.mlp.fc1.weight")
        fc2_weight: torch.Tensor = state_dict.pop(f"encoder.layers.{i}.mlp.fc2.weight")
        fc1_bias = None
        fc2_bias = None
        if not use_gemm and f"encoder.layers.{i}.mlp.fc1.bias" in state_dict:
            fc1_bias = state_dict.pop(f"encoder.layers.{i}.mlp.fc1.bias")
        if not use_gemm and f"encoder.layers.{i}.mlp.fc2.bias" in state_dict:
            fc2_bias = state_dict.pop(f"encoder.layers.{i}.mlp.fc2.bias")

        fc1_weight = fc1_weight.to(get_current_device())
        fc2_weight = fc2_weight.to(get_current_device())
        if fc1_bias is not None:
            fc1_bias = fc1_bias.to(get_current_device())
        if fc2_bias is not None:
            fc2_bias = fc2_bias.to(get_current_device())

        # split weight
        # fc1 weight need to split along dim=0
        fc1_chunks = torch.chunk(fc1_weight, split_size, dim=0)
        # fc2 weight need to split along dim=1
        fc2_chunks = torch.chunk(fc2_weight, split_size, dim=1)
        # fc1 bias need to split along dim=0
        if fc1_bias is not None:
            fc1_bias_chunks = torch.chunk(fc1_bias, split_size, dim=0)
        # fc2 bias do not need to split
        if fc2_bias is not None:
            fc2_bias_chunks = [fc2_bias.clone() / split_size for _ in range(split_size)]

        moe_state = {"fc1": [], "fc2": [], "fc1_bias": [], "fc2_bias": []}

        for k in range(upcycle_size):
            for j in range(split_size):
                expert_i = k * split_size + j
                if ep_rank * num_local_experts <= expert_i < (ep_rank + 1) * num_local_experts:
                    cur_fc1_chunk = fc1_chunks[j]
                    cur_fc2_chunk = fc2_chunks[j]
                    cur_fc1_bias_chunk = fc1_bias_chunks[j] if fc1_bias is not None else None
                    cur_fc2_bias_chunk = fc2_bias_chunks[j] if fc2_bias is not None else None
                    if use_gemm:
                        cur_fc1_chunk = cur_fc1_chunk.T
                        cur_fc2_chunk = cur_fc2_chunk.T

                    if tp_mode == "isp":
                        if ewp_size > 1:
                            cur_fc1_chunk = cur_fc1_chunk.chunk(ewp_size, dim=col_dim)[ewp_rank]
                            cur_fc2_chunk = cur_fc2_chunk.chunk(ewp_size, dim=row_dim)[ewp_rank]
                            if cur_fc1_bias_chunk is not None:
                                cur_fc1_bias_chunk = cur_fc1_bias_chunk.chunk(ewp_size, dim=0)[ewp_rank]
                            if cur_fc2_bias_chunk is not None:
                                cur_fc2_bias_chunk = cur_fc2_bias_chunk.chunk(ewp_size, dim=0)[ewp_rank]
                    else:
                        if etp_size > 1:
                            cur_fc1_chunk = cur_fc1_chunk.chunk(etp_size, dim=col_dim)[etp_rank]
                            cur_fc2_chunk = cur_fc2_chunk.chunk(etp_size, dim=row_dim)[etp_rank]
                            if cur_fc1_bias_chunk is not None:
                                cur_fc1_bias_chunk = cur_fc1_bias_chunk.chunk(etp_size, dim=0)[etp_rank]

                    moe_state["fc1"].append(cur_fc1_chunk)
                    moe_state["fc2"].append(cur_fc2_chunk)

                    if cur_fc1_bias_chunk is not None:
                        moe_state["fc1_bias"].append(cur_fc1_bias_chunk)
                    if cur_fc2_bias_chunk is not None:
                        moe_state["fc2_bias"].append(cur_fc2_bias_chunk)

        if merge_dim0:
            final_moe_state[f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.0.fc1.weight"] = apply_jitter(
                torch.cat(moe_state["fc1"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.0.fc2.weight"] = apply_jitter(
                torch.cat(moe_state["fc2"], dim=0), jitter_scale, generator
            )
        elif use_gemm:
            final_moe_state[f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.0.fc1.weight"] = apply_jitter(
                torch.stack(moe_state["fc1"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.0.fc2.weight"] = apply_jitter(
                torch.stack(moe_state["fc2"], dim=0), jitter_scale, generator
            )
        else:
            for k in range(num_local_experts):
                final_moe_state[
                    f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.{k}.fc1.weight"
                ] = apply_jitter(moe_state["fc1"][k], jitter_scale, generator)
                final_moe_state[
                    f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.{k}.fc2.weight"
                ] = apply_jitter(moe_state["fc2"][k], jitter_scale, generator)
                if len(moe_state["fc1_bias"]) > k:
                    final_moe_state[
                        f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.{k}.fc1.bias"
                    ] = apply_jitter(moe_state["fc1_bias"][k], jitter_scale, generator)
                if len(moe_state["fc2_bias"]) > k:
                    final_moe_state[
                        f"encoder.layers.{i}.mlp.moe_layer.experts.wrapped_experts.{k}.fc2.bias"
                    ] = apply_jitter(moe_state["fc2_bias"][k], jitter_scale, generator)
        del moe_state

        # shared experts
        repeat_num = num_shared_experts // split_size
        mod = num_shared_experts % split_size
        shared_fc1_weight = fc1_chunks * repeat_num + fc1_chunks[:mod]
        shared_fc2_weight = fc2_chunks * repeat_num + fc2_chunks[:mod]
        if fc1_bias is not None:
            shared_fc1_bias = fc1_bias_chunks * repeat_num + fc1_bias_chunks[:mod]
        if fc2_bias is not None:
            shared_fc2_bias = fc2_bias

        shared_fc1_weight = torch.cat(shared_fc1_weight, dim=0)
        shared_fc2_weight = torch.cat(shared_fc2_weight, dim=1)
        if fc1_bias is not None:
            shared_fc1_bias = torch.cat(shared_fc1_bias, dim=0)

        chunk_size = wp_size if tp_mode == "isp" else tp_size
        local_rank = wp_rank if tp_mode == "isp" else tp_rank
        final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc1.weight"] = torch.chunk(
            shared_fc1_weight, chunk_size, dim=col_dim
        )[local_rank]
        final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc2.weight"] = torch.chunk(
            shared_fc2_weight, chunk_size, dim=row_dim
        )[local_rank]
        if fc1_bias is not None:
            final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc1.bias"] = torch.chunk(
                shared_fc1_bias, chunk_size, dim=0
            )[local_rank]
        if fc2_bias is not None:
            if tp_mode == "isp":
                final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc2.bias"] = torch.chunk(
                    shared_fc2_bias, chunk_size, dim=0
                )[local_rank]
            else:
                if local_rank == 0:
                    final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc2.bias"] = shared_fc2_bias.clone()
                else:
                    final_moe_state[f"encoder.layers.{i}.mlp.residual_mlp.fc2.bias"] = None

    return final_moe_state


def load_pretrained_llm_moe_offline(model, model_args, model_path):
    language_model_state_dict = model.state_dict()

    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)
    tp_mode = gpc.config.parallel.tensor["mode"]
    if tp_mode == "isp":
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)

    experts_ckpt_path = None
    shared_experts_ckpt_path = None

    if model_args.moe_kwargs.get("num_shared_experts", 0) > 0:
        if tp_mode == "isp":
            shared_experts_ckpt_path = os.path.join(model_path, f"model_shared_experts_wp{wp_rank}.pt")
        else:
            shared_experts_ckpt_path = os.path.join(model_path, f"model_shared_experts_tp{tp_rank}.pt")

    moe_type = model_args.moe_kwargs.moe_type
    num_local_experts = model_args.moe_kwargs.num_experts // ep_size
    use_gemm = True
    if moe_type == "GShard" and model_args.moe_layer_kwargs.get("use_grouped_mlp", False):
        use_gemm = False

    ckpt_str_prefix = None
    num_local_wrapped_experts = 1 if use_gemm else num_local_experts
    jitter_scale = model_args.moe_kwargs.get("param_jitter_epsilon", 0.0)
    generator = torch.Generator()
    generator.manual_seed(666)  # 保证进程间加的jitter是一致的

    for i in range(gpc.config.model.num_layers_for_pp):
        layer_idx = i + model.first_layer
        moe_str_prefix = "feed_forward.moe_layer.experts.wrapped_experts"
        for j in range(num_local_wrapped_experts):
            global_expert_id = ep_rank * num_local_wrapped_experts + j
            if tp_mode == "isp":
                experts_ckpt_path = os.path.join(
                    model_path, f"model_moe_layer{layer_idx}_expert{global_expert_id}_wp{ewp_rank}.pt"
                )
            else:
                experts_ckpt_path = os.path.join(
                    model_path, f"model_moe_layer{layer_idx}_expert{global_expert_id}_tp0.pt"
                )

            moe_state = torch.load(experts_ckpt_path)
            if not ckpt_str_prefix:
                one_key = list(moe_state.keys())[0]
                ckpt_str_prefix = one_key.split(".")[0]
            for w_i in ["w1", "w2", "w3"]:
                language_model_state_dict[f"layers.{i}.{moe_str_prefix}.{j}.{w_i}.weight"] = apply_jitter(
                    moe_state.pop(
                        f"{ckpt_str_prefix}.layers.{layer_idx}.{moe_str_prefix}.{global_expert_id}.{w_i}.weight"
                    ),
                    jitter_scale,
                    generator,
                )
            del moe_state

        if shared_experts_ckpt_path:
            shared_experts_state = torch.load(shared_experts_ckpt_path)
            for w_i in ["w1", "w2", "w3"]:
                language_model_state_dict[
                    f"layers.{i}.feed_forward.residual_mlp.{w_i}.weight"
                ] = shared_experts_state.pop(
                    f"{ckpt_str_prefix}.layers.{layer_idx}.feed_forward.residual_mlp.{w_i}.weight"
                )
            del shared_experts_state

    message = model.load_state_dict(language_model_state_dict, strict=False)

    return message


def load_pretrained_llm_moe_online(model, state_dict, model_args, from_hf: bool = True):
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    tp_mode = gpc.config.parallel.tensor["mode"]
    ep_size = gpc.get_world_size(ParallelMode.EXPERT)
    ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)

    row_dim = 0 if tp_mode == "isp" else 1
    col_dim = 0

    if tp_mode == "isp":
        ewp_size = gpc.get_world_size(ParallelMode.EXPERT_WEIGHT)
        ewp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)

    jitter_scale = model_args.moe_kwargs.get("param_jitter_epsilon", 0.0)
    generator = torch.Generator()
    generator.manual_seed(666)  # 保证进程间加的jitter是一致的

    merge_dim0 = False
    use_gemm = True
    moe_type = model_args.moe_kwargs.moe_type
    num_experts = model_args.moe_kwargs.num_experts
    num_local_experts = num_experts // ep_size
    num_shared_experts = model_args.moe_kwargs.get("num_shared_experts", 0)
    if moe_type == "GShard" and model_args.moe_layer_kwargs.get("use_grouped_mlp", True):
        use_gemm = False

    if (use_gemm and tp_mode == "isp") or moe_type == "MegaBlock-D":
        merge_dim0 = True

    split_size = model_args.moe_kwargs.get("split_size", model_args.moe_kwargs.top_k)
    upcycle_size = num_experts // split_size

    final_moe_state = {}
    for i in range(0, gpc.config.model.num_layers_for_pp):
        if from_hf:
            w1_weight: torch.Tensor = state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w1.weight")
            w3_weight: torch.Tensor = state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w3.weight")
            w2_weight: torch.Tensor = state_dict.pop(f"model.layers.{i+model.first_layer}.feed_forward.w2.weight")
        else:
            w1_weight: torch.Tensor = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w1.weight"
            )
            w3_weight: torch.Tensor = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w3.weight"
            )
            w2_weight: torch.Tensor = state_dict.pop(
                f"language_model.layers.{i+model.first_layer}.feed_forward.w2.weight"
            )

        w1_weight = w1_weight.to(get_current_device())
        w3_weight = w3_weight.to(get_current_device())
        w2_weight = w2_weight.to(get_current_device())

        # gather weight
        if from_hf is False:
            gather_size = wp_size if tp_mode == "isp" else tp_size
            process_group = (
                gpc.get_group(ParallelMode.WEIGHT) if tp_mode == "isp" else gpc.get_group(ParallelMode.TENSOR)
            )

            if gather_size > 1:
                # w1
                gather_w1_weight, w1_handle = all_gather_raw(
                    w1_weight, process_group, async_op=True, gather_dim=col_dim
                )

                # w3
                gather_w3_weight, w3_handle = all_gather_raw(
                    w3_weight, process_group, async_op=True, gather_dim=col_dim
                )

                # w2
                gather_w2_weight, _ = all_gather_raw(w2_weight, process_group, async_op=False, gather_dim=row_dim)

                w1_handle.wait()
                w3_handle.wait()

            else:
                gather_w1_weight = w1_weight
                gather_w3_weight = w3_weight
                gather_w2_weight = w2_weight
        else:
            gather_w1_weight = w1_weight
            gather_w3_weight = w3_weight
            gather_w2_weight = w2_weight

        # split weight
        w1_chunks = torch.chunk(gather_w1_weight, split_size, dim=0)
        w3_chunks = torch.chunk(gather_w3_weight, split_size, dim=0)
        w2_chunks = torch.chunk(gather_w2_weight, split_size, dim=1)

        moe_state = {"w1": [], "w2": [], "w3": []}
        for k in range(upcycle_size):
            for j in range(split_size):
                expert_i = k * split_size + j
                if ep_rank * num_local_experts <= expert_i < (ep_rank + 1) * num_local_experts:
                    cur_w1_chunk = w1_chunks[j]
                    cur_w3_chunk = w3_chunks[j]
                    cur_w2_chunk = w2_chunks[j]
                    if use_gemm:
                        cur_w1_chunk = cur_w1_chunk.T
                        cur_w3_chunk = cur_w3_chunk.T
                        cur_w2_chunk = cur_w2_chunk.T

                    if tp_mode == "isp" and ewp_size > 1:
                        cur_w1_chunk = cur_w1_chunk.chunk(ewp_size, dim=col_dim)[ewp_rank]
                        cur_w3_chunk = cur_w3_chunk.chunk(ewp_size, dim=col_dim)[ewp_rank]
                        cur_w2_chunk = cur_w2_chunk.chunk(ewp_size, dim=row_dim)[ewp_rank]

                    moe_state["w1"].append(cur_w1_chunk)
                    moe_state["w3"].append(cur_w3_chunk)
                    moe_state["w2"].append(cur_w2_chunk)

        if merge_dim0:
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w1.weight"] = apply_jitter(
                torch.cat(moe_state["w1"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w3.weight"] = apply_jitter(
                torch.cat(moe_state["w3"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w2.weight"] = apply_jitter(
                torch.cat(moe_state["w2"], dim=0), jitter_scale, generator
            )
        elif use_gemm:
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w1.weight"] = apply_jitter(
                torch.stack(moe_state["w1"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w3.weight"] = apply_jitter(
                torch.stack(moe_state["w3"], dim=0), jitter_scale, generator
            )
            final_moe_state[f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.0.w2.weight"] = apply_jitter(
                torch.stack(moe_state["w2"], dim=0), jitter_scale, generator
            )
        else:
            for k in range(num_local_experts):
                final_moe_state[
                    f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.{k}.w1.weight"
                ] = apply_jitter(moe_state["w1"][k], jitter_scale, generator)
                final_moe_state[
                    f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.{k}.w3.weight"
                ] = apply_jitter(moe_state["w3"][k], jitter_scale, generator)
                final_moe_state[
                    f"layers.{i}.feed_forward.moe_layer.experts.wrapped_experts.{k}.w2.weight"
                ] = apply_jitter(moe_state["w2"][k], jitter_scale, generator)
        del moe_state

        # shared experts
        repeat_num = num_shared_experts // split_size
        mod = num_shared_experts % split_size
        shared_w1_weight = w1_chunks * repeat_num + w1_chunks[:mod]
        shared_w3_weight = w3_chunks * repeat_num + w3_chunks[:mod]
        shared_w2_weight = w2_chunks * repeat_num + w2_chunks[:mod]

        shared_w1_weight = torch.cat(shared_w1_weight, dim=0)
        shared_w3_weight = torch.cat(shared_w3_weight, dim=0)
        shared_w2_weight = torch.cat(shared_w2_weight, dim=1)

        chunk_size = wp_size if tp_mode == "isp" else tp_size
        local_rank = wp_rank if tp_mode == "isp" else tp_rank
        final_moe_state[f"layers.{i}.feed_forward.residual_mlp.w1.weight"] = torch.chunk(
            shared_w1_weight, chunk_size, dim=col_dim
        )[local_rank]
        final_moe_state[f"layers.{i}.feed_forward.residual_mlp.w3.weight"] = torch.chunk(
            shared_w3_weight, chunk_size, dim=col_dim
        )[local_rank]
        final_moe_state[f"layers.{i}.feed_forward.residual_mlp.w2.weight"] = torch.chunk(
            shared_w2_weight, chunk_size, dim=row_dim
        )[local_rank]

    return final_moe_state


def load_pretrained_model(model, vit_args, llm_args, model_path):
    # load files
    state_dict = _load_hf_checkpoint(model_path)

    state_dict_vit, state_dict_llm, state_dict_mlp, state_dict_fm_modules = _split_state_dict(state_dict)
    state_dict = None

    use_moe = gpc.config.model.use_moe
    moe_location = gpc.config.model.moe_location

    # for generation pre-training, we will randomly initialize the vision embeddings
    # if len(state_dict_fm_modules) > 0:
    if hasattr(model, "vision_model"):
        has_vit_moe = use_moe and moe_location == "vision"
        message_vit = _load_pretrained_hf_vit(
            model.vision_model, vit_args, state_dict_vit, has_moe=has_vit_moe
        )  # pylint:disable=W0612

        if gpc.is_rank_for_log():
            logger.info(f"load vit: {message_vit}")

    if hasattr(model, "mlp1"):
        message_mlp = load_pretrained_mlp(model.mlp1, state_dict=state_dict_mlp)  # pylint:disable=W0612
        if gpc.is_rank_for_log():
            logger.info(f"load mlp1: {message_mlp}")

    if hasattr(model, "language_model"):
        has_llm_moe = use_moe and moe_location == "llm"
        message_llm = _load_pretrained_hf_llm(
            model=model.language_model, model_args=llm_args, state_dict=state_dict_llm, has_moe=has_llm_moe
        )
        if gpc.is_rank_for_log():
            logger.info(f"load llm: {message_llm}")

    if hasattr(model, "fm_modules"):
        message_fm_modules = load_pretrained_fm_modules(model.fm_modules, state_dict=state_dict_fm_modules)  # pylint:disable=W0612
        if gpc.is_rank_for_log():
            logger.info(f"load fm_modules: {message_fm_modules}")

    sensenovalm_accelerator.empty_cache()


