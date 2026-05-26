# Copyright (c) OpenMMLab. All rights reserved.
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original portions retain their Apache-2.0 license.
# Upstream reference: https://github.com/open-mmlab
import types
from contextlib import contextmanager
from enum import Enum, IntEnum
from functools import update_wrapper
from typing import Callable, Tuple

import torch


@contextmanager
def read_base():
    """Context manager to mark the base config.

    The pure Python-style configuration file allows you to use the import
    syntax. However, it is important to note that you need to import the base
    configuration file within the context of ``read_base``, and import other
    dependencies outside of it.

    You can see more usage of Python-style configuration in the `tutorial`_

    .. _tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta  # pylint: disable=line-too-long
    """  # noqa: E501
    yield


class QKVPackType(IntEnum):
    QKVPACKED = 2
    KVPACKED = 3
    QKVSPLITED = 4

    def __str__(self) -> str:
        return str(self.value)


class CuSeqlenType(Enum):
    With = True
    WithOut = False

    def __str__(self) -> str:
        return str(self.value)


class ModelType(Enum):
    """
    model types supported in InternEvo
    """

    SENSENOVALM = 1
    SENSENOVALM2_PUBLIC = 2
    LLAMA2 = 3
    SENSENOVALM_MoE = 4
    LLAVA = 5
    QWEN2 = 6
    BAICHUAN2 = 7
    GEMMA = 8
    QWEN2MOE = 9
    MIXTRALMOE = 10
    SENSENOVALM2_MoE = 11
    DEEPSEEK3_MoE = 12
    QWEN3_MoEMoT = 14


class DataType(Enum):
    streaming = 1
    tokenized = 2
    megatron = 3
    mocked = 4
    multimodal_streaming = 5
    multimodal_packed_streaming = 6


class TensorParallelMode(Enum):
    mtp = 1
    msp = 2
    fsp = 3
    isp = 4


class ActivationType(Enum):
    swiglu = 1
    gelu = 2


def check_attention_argument(*args, **kwargs) -> str:
    # self, qkv, ...
    # self, q, kv, ....
    # self, q, k, v, ...
    # self, qkv, cu_seqlens, max_seqlen, ...
    # self, q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...
    # self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...
    def __qkv_checker(num_args: int):
        if num_args < 2:
            return "qkv" in kwargs
        else:
            return len(args[1].shape) == 5

    def __kv_checker(num_args: int):
        if num_args < 3:
            return "kv" in kwargs
        else:
            return len(args[2].shape) == 5

    def __cu_seqlens_checker(args, check_idx: int):
        num_args = len(args)
        if num_args < (check_idx + 1):
            if check_idx == 2:
                return "cu_seqlens" in kwargs and kwargs["cu_seqlens"] is not None
            else:
                return "cu_seqlens_q" in kwargs and kwargs["cu_seqlens_q"] is not None
        else:
            return isinstance(args[check_idx], torch.Tensor)

    if __qkv_checker(len(args)):
        # qkv packed, and we should check cu_seqlens with index 2
        qkv_pack_type = int(QKVPackType.QKVPACKED)
    elif __kv_checker(len(args)):
        # kv packed, and we should check cu_seqlens with index 3
        qkv_pack_type = int(QKVPackType.KVPACKED)
    else:
        # qkv splited, and we should check cu_seqlens with index 4
        qkv_pack_type = int(QKVPackType.QKVSPLITED)

    with_cu_seqlens = __cu_seqlens_checker(args, qkv_pack_type)

    return str(qkv_pack_type), str(with_cu_seqlens)


def params_dispatch_with_condition(condition: Callable, func: Callable = None):

    if func is None:
        # create a params dispatch wrapper
        return lambda f: params_dispatch_with_condition(condition, f)

    registry = {}
    funcname = getattr(func, "__name__", "params_dispatch_with_condition function")

    def dispatch(_type: str) -> Callable:
        return registry[_type]

    def register(conditions: Tuple[str], func: Callable = None) -> None:
        if func is None:
            # create a register wrapper
            return lambda f: register(conditions, f)

        _type = "-".join(conditions)

        assert _type not in registry, f"Repeatedly register dispatch functions for pattern {_type}"

        registry[_type] = func

        return func

    def wrapper(*args, **kwargs):
        if not args:
            raise TypeError(f"{funcname} requires at least " "1 positional argument")

        _type = "-".join(condition(*args, **kwargs))

        return dispatch(_type)(*args, **kwargs)

    registry[""] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    update_wrapper(wrapper, func)
    return wrapper
