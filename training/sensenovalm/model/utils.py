# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Union

from torch import nn
from tqdm import tqdm

from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.core.naive_amp import unwrap_naive_amp
from sensenovalm.model.modules.mha import MHA
from sensenovalm.utils.storage_manager import get_fns, llm_load
from sensenovalm.utils.utils import TensorParallelMode

_T = TypeVar("_T")


def sensenovalm1_mha_pre_load_convert(
    model: MHA, state_dict: Dict, prefix: str, *args, **kwargs  # pylint: disable=W0613
) -> None:
    if f"{prefix}wqkv.weight" not in state_dict and f"{prefix}Wqkv.weight" in state_dict:
        state_dict[f"{prefix}wqkv.weight"] = state_dict.pop(f"{prefix}Wqkv.weight")

    if f"{prefix}wqkv.bias" not in state_dict and f"{prefix}Wqkv.bias" in state_dict:
        state_dict[f"{prefix}wqkv.bias"] = state_dict.pop(f"{prefix}Wqkv.bias")


def sensenovalm1_mha_save_convert(
    model: MHA, state_dict: Dict, prefix: str, *args, **kwargs  # pylint: disable=W0613
) -> None:
    state_dict[f"{prefix}Wqkv.weight"] = state_dict.pop(f"{prefix}wqkv.weight")

    if f"{prefix}wqkv.bias" in state_dict:
        state_dict[f"{prefix}Wqkv.bias"] = state_dict.pop(f"{prefix}wqkv.bias")


def convert_attn_kwargs_to_args(kwargs) -> List[Any]:
    inference_params = kwargs.get("inference_params", None)
    cu_seqlens = kwargs.get("cu_seqlens", None)
    indexes = kwargs.get("indexes", None)
    max_seqlen = kwargs.get("max_seqlen", None)

    # NOTE:
    flex_mask=kwargs.get("flex_mask", None)
    padlen=kwargs.get("padlen", None)

    return (inference_params, cu_seqlens, indexes, max_seqlen, flex_mask, padlen)


def convert_attn_args_to_kwargs(args, kwargs) -> Dict[str, Any]:
    if len(args) == 0:
        return kwargs

    # NOTE:
    assert len(args) == 6, "args must be generate by convert_attn_kwargs_to_args function"

    if args[0] is not None:
        assert "inference_params" not in kwargs, "repeated 'inference_params' argument exists both in args and kwargs"
        kwargs["inference_params"] = args[0]
    if args[1] is not None:
        assert "cu_seqlens" not in kwargs, "repeated 'cu_seqlens' argument exists both in args and kwargs"
        kwargs["cu_seqlens"] = args[1]
    if args[2] is not None:
        assert "indexes" not in kwargs, "repeated 'indexes' argument exists both in args and kwargs"
        kwargs["indexes"] = args[2]
    if args[3] is not None:
        assert "max_seqlen" not in kwargs, "repeated 'max_seqlen' argument exists both in args and kwargs"
        kwargs["max_seqlen"] = args[3]
    
    # NOTE:
    if args[4] is not None:
        assert "flex_mask" not in kwargs, "repeated 'flex_mask' argument exists both in args and kwargs"
        kwargs["flex_mask"] = args[4]
    if args[5] is not None:
        assert "padlen" not in kwargs, "repeated 'padlen' argument exists both in args and kwargs"
        kwargs["padlen"] = args[5]

    return kwargs


def _find_max_tp_pp(names):
    ckpt_names = []
    for name in names:
        if name.startswith("model_t") and not name.endswith("md5"):
            # _t: avoid conflictint with model_config.pt
            ckpt_names.append(name)

    max_tp, max_pp = -1, -1
    for ckpt in ckpt_names:
        _, tp, pp = os.path.splitext(ckpt)[0].split("_")
        max_tp = max(max_tp, int(tp[2:]) + 1)
        max_pp = max(max_pp, int(pp[2:]) + 1)

    return max_tp, max_pp


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


def load_src_states(src):
    ckpt_names = get_fns(src)
    if gpc.config.parallel.tensor.mode == TensorParallelMode.isp.name:
        max_wp, max_pp = _find_max_wp_pp(ckpt_names)
        # 2-d array wp_rank, pp_rank
        states = [[None for _ in range(max_pp)] for __ in range(max_wp)]
        for wp in tqdm(range(max_wp)):
            for pp in tqdm(range(max_pp)):
                ckpt_name = os.path.join(src, f"model_wp{wp}_pp{pp}.pt")
                states[wp][pp] = llm_load(ckpt_name, map_location="cpu")
    else:
        max_tp, max_pp = _find_max_tp_pp(ckpt_names)
        # 2-d array tp_rank, pp_rank
        states = [[None for _ in range(max_pp)] for __ in range(max_tp)]
        for tp in tqdm(range(max_tp)):
            for pp in tqdm(range(max_pp)):
                ckpt_name = os.path.join(src, f"model_tp{tp}_pp{pp}.pt")
                states[tp][pp] = llm_load(ckpt_name, map_location="cpu")
    return states


def merge_pp_src_states(states):
    merged_states = []
    for tp_state in tqdm(states):
        layer_shift = 0
        shifted_state = {}
        # shift key
        for tp_pp_state in tp_state:
            _layer_shift = 0
            keys = list(tp_pp_state.keys())
            for key in keys:
                if key.endswith(".inv_freq"):
                    continue
                match = re.search(r"\.\d+\.", key)
                name = key
                if match is not None:
                    # layers
                    s, e = match.span()
                    layer_idx = int(key[s + 1 : e - 1]) + layer_shift
                    _layer_shift = max(_layer_shift, int(key[s + 1 : e - 1]))
                    name = key[:s] + f".{layer_idx}." + key[e:]
                if name.startswith("model."):
                    name = name[6:]
                shifted_state[name] = tp_pp_state[key]
            layer_shift += _layer_shift + 1
        merged_states.append(shifted_state)
    return merged_states


def _submodule_filter(model: Union[nn.Module, nn.ModuleList], target_cls: Union[_T, Tuple[_T]]) -> Iterable[_T]:
    for _chunk in unwrap_naive_amp(model):
        for _module in _chunk.modules():
            if not isinstance(_module, target_cls):
                continue

            yield _module
