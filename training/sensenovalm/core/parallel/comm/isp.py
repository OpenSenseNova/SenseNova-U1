#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from DeepSpeed (Microsoft, Apache-2.0).
# Upstream references:
#   https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
# -*- encoding: utf-8 -*-
"""
communication for isp parallel.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch import nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.naive_amp import NaiveAMPModel
from sensenovalm.core.parallel.comm.utils import (
    DUMMY_HANDLE_CONST,
    AsyncCommHandle,
    _gather,
    _split,
    all_gather_raw,
    apply_to_tensors_only,
    expandKVPacked,
    reduce_scatter_raw,
)
from sensenovalm.model.modules.embedding import Embedding1D
from sensenovalm.model.modules.linear import ParallelLinearWithCommExt
from sensenovalm.model.modules.utils import is_moe_param
from sensenovalm.utils.common import SchedulerHook, UniqueChainMap, get_current_device
from sensenovalm.utils.utils import (
    CuSeqlenType,
    QKVPackType,
    TensorParallelMode,
    check_attention_argument,
    params_dispatch_with_condition,
)


# not really useful, only for code hint.
class WPCommunicator(ABC):
    """
    Common communicator interface for weight parallel
    """

    @abstractmethod
    def communication_mode(self) -> str:
        """
        communication mode of communictor
        """
        pass

    @abstractmethod
    def weight_hook(self, tensor: torch.Tensor, async_op: bool = False, **kwargs) -> torch.Tensor:
        """
        communication for weight when forward/backward.
        """
        pass

    @abstractmethod
    def grad_hook(self, tensor: torch.Tensor, async_op: bool = False, **kwargs) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for grad when backward.
        """
        pass


class HeadWeightParallelCommunicator(WPCommunicator):
    """
    Weight parallel communicator for Head module.
    """

    def __init__(
        self,
        weight_process_group: dist.ProcessGroup = None,
        seq_process_group: dist.ProcessGroup = None,
        retain_out_sharded: bool = True,
    ) -> None:
        self.weight_process_group = weight_process_group
        self.seq_process_group = seq_process_group
        self._seq_parallel_mode = ParallelMode.TENSOR
        self._seq_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self._retain_out_sharded = retain_out_sharded
        self._seq_dim = 1
        self._hid_dim = 2

    def communication_mode(self) -> str:
        return "wp"

    def weight_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,  # pylint: disable=W0613
        is_bias: bool = False,  # pylint: disable=W0613
    ) -> torch.Tensor:
        if dist.get_world_size(self.weight_process_group) <= 1:
            return tensor

        result, _ = all_gather_raw(tensor, self.weight_process_group, async_op=async_op)
        return result

    def grad_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,  # pylint: disable=W0613
        reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
        is_bias: bool = False,  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        if dist.get_world_size(self.weight_process_group) <= 1:
            return tensor, DUMMY_HANDLE_CONST

        result, handle = reduce_scatter_raw(tensor, self.weight_process_group, op=reduce_op, async_op=async_op)
        return result, handle

        # rewrite grad_output communication hook

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        split grad_output if retain_out_sharded is False.
        """

        # gather hidden_states dim and split seq dim when parallel_output is True
        if self._retain_out_sharded:
            if self._seq_world_size <= 1:
                return grad_output, DUMMY_HANDLE_CONST
            else:
                _seq_splited_list = [
                    t.contiguous() for t in torch.tensor_split(grad_output, self._seq_world_size, dim=self._seq_dim)
                ]
                output_list = [torch.empty_like(_seq_splited_list[0]) for _ in range(self._seq_world_size)]
                dist.all_to_all(output_list, _seq_splited_list, group=self.seq_process_group)
                grad_output = torch.cat(output_list, dim=self._hid_dim).contiguous()
                return grad_output, DUMMY_HANDLE_CONST
        # split seq dim when parallel_output is False
        else:
            if self._seq_world_size <= 1:
                return grad_output, DUMMY_HANDLE_CONST
            else:
                return _split(grad_output, parallel_mode=self._seq_parallel_mode, dim=self._seq_dim), DUMMY_HANDLE_CONST

    # rewrite ouput communication hook
    def output_hook(
        self, output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather output for head layer if retain_out_sharded is False.
        """

        # gather seq dim and split hidden_states dim when parallel_output is True
        if self._retain_out_sharded:
            if self._seq_world_size <= 1:
                return output, DUMMY_HANDLE_CONST
            else:
                _hid_splited_list = [
                    t.contiguous() for t in torch.tensor_split(output, self._seq_world_size, dim=self._hid_dim)
                ]
                output_list = [torch.empty_like(_hid_splited_list[0]) for _ in range(self._seq_world_size)]
                dist.all_to_all(output_list, _hid_splited_list, group=self.seq_process_group)
                output = torch.cat(output_list, dim=self._seq_dim).contiguous()
                return output, DUMMY_HANDLE_CONST
        # gather seq dim when parallel_output is False
        else:
            if self._seq_world_size <= 1:
                return output, DUMMY_HANDLE_CONST
            else:
                return _gather(output, parallel_mode=self._seq_parallel_mode, dim=self._seq_dim), DUMMY_HANDLE_CONST


class EmbeddingWeightParallelCommunicator:
    """
    Weight parallel communicator for embedding layer.
    """

    def __init__(self, parallel_mode: ParallelMode, freeze: bool = False) -> None:
        self.parallel_mode = parallel_mode
        self.gather_dim = 0

        self._cur_micro_step = 0
        self._num_micro_step = gpc.config.data.micro_num

        self._freeze = freeze

    def register_module_hook(self, module: Embedding1D) -> None:
        assert isinstance(module, Embedding1D), "Embbeding weight parallel communicator is only support Embedding1D"

        module.weight.evo_tensor = None
        self.gather_dim = 0 if module.vocab_parallel else 1

        class PreModuleWrapper(torch.autograd.Function):
            """
            Wrapper pre module to prefetch module weight for forward pass.
            """

            @staticmethod
            def forward(ctx, inputs: torch.Tensor):  # pylint: disable=W0613
                if module.weight.evo_tensor is None:
                    module.weight.evo_tensor = module.weight.data

                module.weight.data = _gather(module.weight, self.parallel_mode, dim=self.gather_dim)
                inputs = inputs.detach()
                return inputs

            @staticmethod
            def backward(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
                # since input of embedding is int64 dtype, requires_grad=False, the backward fn may not be called
                module.weight.data = module.weight.evo_tensor
                return grad_input

        class PostModuleWrapper(torch.autograd.Function):
            """
            Wrapper post module to prefetch module weight for backward pass.
            """

            @staticmethod
            def forward(ctx, output: torch.Tensor):  # pylint: disable=W0613
                module.weight.data = module.weight.evo_tensor
                output = output.detach()
                return output

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
                module.weight.data = _gather(module.weight, self.parallel_mode, dim=self.gather_dim)
                return grad_output

        def _pre_forward_hook(module, inputs):  # pylint: disable=W0613
            return apply_to_tensors_only(PreModuleWrapper.apply, inputs)

        def _post_forward_hook(module, inputs, output):  # pylint: disable=W0613
            return apply_to_tensors_only(PostModuleWrapper.apply, output)

        def _post_backward_hook(module, *args):  # pylint: disable=W0613
            module.weight.data = module.weight.evo_tensor

        module.register_forward_pre_hook(_pre_forward_hook)
        module.register_forward_hook(_post_forward_hook)
        module.register_full_backward_hook(_post_backward_hook)

        if self._freeze is False:
            module.weight.register_post_accumulate_grad_hook(self.grad_reduce_hook)

    def grad_reduce_hook(self, param: torch.Tensor):

        _grad, _ = reduce_scatter_raw(
            param.grad, gpc.get_group(self.parallel_mode), op=dist.ReduceOp.SUM, reduce_dim=self.gather_dim
        )
        if param.evo_tensor.grad is None:
            param.evo_tensor.grad = _grad
        else:
            param.evo_tensor.grad += _grad

        param.data = param.evo_tensor
        param.grad = None

        self._cur_micro_step += 1
        if self._cur_micro_step == self._num_micro_step:
            param.grad = param.evo_tensor.grad
            param.evo_tensor.grad = None
            self._cur_micro_step = 0


class ISPCommModelConfig:
    """
    model config for isp communicator.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.half,
        device: torch.device = None,
        activation_checkpointing: float = 0.0,
        pure_llm: bool = False,
    ) -> None:
        self.dtype = dtype
        if device is None:
            self.device = get_current_device()
        else:
            self.device = device
        self.activation_checkpointing = activation_checkpointing
        self.pure_llm = pure_llm


class ISPOverlapState:
    """
    Overlap state for isp.
    """

    def __init__(self) -> None:
        self.num_blocks: int = 0
        self.ckpt_block_num: int = 0
        self.ckpt_per_block_num: int = 1
        self.isp_prefetch_launch_module: List[nn.Module] = []
        self.isp_modules: List[nn.Module] = []
        self.index_to_isp_modules: Dict[int, nn.Module] = {}
        self.index_to_block: Dict[int, nn.Module] = {}
        self.module_to_index: Dict[nn.Module, int] = {}
        self.weight_global_handle: Dict[str, Any] = {}
        self.weight_global_output: Dict[str, torch.Tensor] = {}
        self.bias_global_handle: Dict[str, Any] = {}
        self.bias_global_output: Dict[str, torch.Tensor] = {}


class ISPCommunicator(WPCommunicator):
    """
    ISP Communicator for managing the all-gather and reduce_scatter of Intern Sequence Parallel.
    """

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        model_conf: ISPCommModelConfig,
        overlap: bool = False,
        process_group: dist.ProcessGroup = None,
        is_moe: bool = False,
        early_reduce_scatter_release: bool = True,
    ) -> None:
        self.process_group = process_group
        self.overlap = overlap
        self.model_conf = model_conf
        self.is_moe = is_moe
        self.is_forward = True
        self.reduce_scatter_handlers = {}
        self._forward_prefetch_prerequisites = []
        self._forward_overlap_per = self._get_forward_overlap_granularity()
        self._launch_before_module = self._get_launch_before_module()
        self._zero_const_pool = {}

        self._enable_early_reduce_scatter_release = early_reduce_scatter_release
        self._early_prev_layer_rs_handles = []
        self._early_curr_layer_rs_handles = []

        # mlp module with isp linear but not overlap
        self._cid_to_mlp_isp_no_overlap: Dict[int, list] = {}

        # real overlap state for each chunk.
        self._overlap_states: Dict[int, dict] = {}

        # inner interface variables of overlap state.
        self._num_blocks = None
        self._ckpt_block_num = None
        self._ckpt_per_block_num = None
        self._isp_prefetch_launch_module = None
        self._isp_modules = None
        # key: isp module; value: module global all-gather op handle
        self._weight_global_handle = None
        # key: isp module; value: module bias global all-gather op handle
        self._bias_global_handle = None
        # key: isp module; value: module global weight after all-gather op
        self._weight_global_output = None
        # key: isp module; value: module bias global weight after all-gather op
        self._bias_global_output = None
        # key: isp module; value: transformer block index
        self._module_to_index = None
        # key: transformer block index; value: isp modules
        self._index_to_isp_modules = None
        # key: transformer block index; value: transformer block
        self._index_to_block = None
        # current mlp module with isp linear but not overlap
        self._mlp_isp_no_overlap = []

        # init overlap states if necessary.
        if self.overlap:
            # just want to share same for loop for modulelist and module.
            model = model if isinstance(model, nn.ModuleList) else [model]

            # build overlap states for every chunk.
            for chunk_id, chunk in enumerate(model):
                if isinstance(chunk, NaiveAMPModel):
                    chunk = chunk.model
                self._parse_model_structure(chunk_id, chunk)

            # register overlap hooks for every chunk.
            for chunk_id, chunk in enumerate(model):
                if isinstance(chunk, NaiveAMPModel):
                    chunk = chunk.model

                if hasattr(chunk, "vision_model") and not chunk.vision_model.encoder.is_dummy:
                    # register overlap hooks for vision layers
                    is_llm = False
                    self.switch_current_model_chunk(chunk_id, is_llm)
                    self._register_sync_parameters_hook()

                if hasattr(chunk, "language_model"):
                    # register overlap hooks for llm layers
                    is_llm = True
                    self.switch_current_model_chunk(chunk_id, is_llm)
                    self._register_sync_parameters_hook()

            # switch to chunk 0 and vision layers at first.
            _is_llm = False
            if self.model_conf.pure_llm or gpc.get_local_rank(ParallelMode.PIPELINE) >= 1:
                _is_llm = True
            self.switch_current_model_chunk(0, _is_llm)

    def _get_launch_before_module(self):
        if self.is_moe is True:
            _launch_before = gpc.config.parallel.expert_weight.get("launch_allgather_before", "wo")
        else:
            _launch_before = gpc.config.parallel.weight.get("launch_allgather_before", "wo")
        if _launch_before == "wqkv":
            return ["wqkv", "Wqkv", "qkv", "q_a_proj", "q_proj"]
        elif _launch_before == "attn":
            return ["attn"]
        elif _launch_before == "wo":
            return ["out_proj", "wo", "proj"]
        elif _launch_before == "w1":
            return ["w1", "fused_w1_w3"]
        else:
            assert False, "launch module should be in ['wqkv', 'attn', 'wo', 'w1']"

    def _get_forward_overlap_granularity(self):
        if self.is_moe is True:
            _overlap_granularity = gpc.config.parallel.expert_weight.get("forward_overlap_per", "layer")
        else:
            _overlap_granularity = gpc.config.parallel.weight.get("forward_overlap_per", "layer")
        assert _overlap_granularity in ["module", "layer"]
        return _overlap_granularity

    def _parse_model_structure(self, cid: int, model: nn.Module) -> None:
        self._overlap_states[cid] = {}

        def is_allgather_launch_module(name, module):
            return (
                hasattr(module, "is_attn_cls")
                and getattr(module, "is_attn_cls")
                and self._launch_before_module == ["attn"]
            ) or (name.split(".")[-1] in self._launch_before_module)

        # Important: only works for llama-class models
        # vision model layers
        if hasattr(model, "vision_model") and not model.vision_model.encoder.is_dummy:
            vision_layers = model.vision_model.encoder.layers
            assert isinstance(vision_layers, nn.ModuleList)
        # llm model layers
        if hasattr(model, "language_model"):
            llm_layers = model.language_model.layers
            if hasattr(model.language_model, "mtp"):
                mtp_layers = model.language_model.mtp.mtp_layers
                llm_layers = nn.ModuleList(list(llm_layers) + list(mtp_layers))
            assert isinstance(llm_layers, nn.ModuleList)

        # register pre-forward hook to switch current model overlap states to llm
        def _pre_llm_forward_hook_for_module(module: nn.Module, *args):  # pylint: disable=W0613
            self.switch_current_model_chunk(cid, True)

        # register pre-backward hook to switch current model overlap states to vision
        def _pre_vision_backward_hook_for_module(module: nn.Module, *args):  # pylint: disable=W0613
            self.switch_current_model_chunk(cid, False)

        self._cid_to_mlp_isp_no_overlap[cid] = []
        if hasattr(model, "mlp1"):
            _before_llm_module = model.mlp1
            if hasattr(model, "language_model"):
                _before_llm_module.register_forward_pre_hook(_pre_llm_forward_hook_for_module)
            _before_llm_module.register_full_backward_pre_hook(_pre_vision_backward_hook_for_module)

            assert isinstance(model.mlp1, nn.Sequential)
            for _mlp_module in model.mlp1:
                if isinstance(_mlp_module, (ParallelLinearWithCommExt)):
                    self._cid_to_mlp_isp_no_overlap[cid].append(_mlp_module)
        elif model.vision_model.encoder.is_dummy:
            model.vision_model.register_forward_pre_hook(_pre_llm_forward_hook_for_module)
            model.vision_model.register_full_backward_pre_hook(_pre_vision_backward_hook_for_module)

        def process_model_layers(
            model: nn.ModuleList, activation_checkpointing, ckpt_per_layer_num: int = 1, is_llm: bool = True
        ):
            _overlap_states = ISPOverlapState()
            _overlap_states.ckpt_block_num = int(activation_checkpointing * len(model))
            _overlap_states.ckpt_per_block_num = ckpt_per_layer_num
            if is_llm:
                assert ckpt_per_layer_num == 1

            for idx, block in enumerate(model):
                _overlap_states.index_to_isp_modules[idx] = []
                _overlap_states.index_to_block[idx] = block
                for sub_name, sub in block.named_children():
                    for name, child in sub.named_modules():
                        if is_allgather_launch_module(name, child):
                            _overlap_states.isp_prefetch_launch_module.append(child)
                            _overlap_states.module_to_index[child] = idx
                        if isinstance(child, (ParallelLinearWithCommExt)):
                            if is_moe_param(child.weight) != self.is_moe:
                                continue

                            _overlap_states.module_to_index[child] = idx
                            _overlap_states.isp_modules.append(child)
                            _overlap_states.index_to_isp_modules[idx].append(child)

                            setattr(child, "isp_name", name)

                            if is_llm:
                                full_name = f"{cid}.llm.{idx}.{sub_name}.{name}"
                            else:
                                full_name = f"{cid}.vision.{idx}.{sub_name}.{name}"
                            setattr(
                                child.weight,
                                "isp_reduce_scatter_name",
                                f"{full_name}.weight",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "isp_reduce_scatter_name",
                                    f"{full_name}.bias",
                                )

            _overlap_states.num_blocks = len(_overlap_states.index_to_isp_modules)

            return _overlap_states

        if hasattr(model, "vision_model"):  # if not only train llm
            if not model.vision_model.encoder.is_dummy:
                # process vision model layers
                vision_layers_states = process_model_layers(
                    vision_layers,
                    gpc.config.model.vit_cfg.gradient_checkpointing,
                    gpc.config.model.vit_cfg.checkpoint_per_layer_num,
                    is_llm=False,
                )
            else:
                vision_layers_states = ISPOverlapState()
                vision_layers_states.ckpt_block_num = 0
                vision_layers_states.ckpt_per_block_num = gpc.config.model.vit_cfg.checkpoint_per_layer_num
                vision_layers_states.num_blocks = 0

            self._overlap_states[cid]["vision"] = vision_layers_states

        # process llm model layers
        if hasattr(model, "language_model"):
            llm_layers_states = process_model_layers(
                llm_layers,
                gpc.config.model.checkpoint,
                gpc.config.model.checkpoint_per_layer_num,
                is_llm=True,
            )
            self._overlap_states[cid]["llm"] = llm_layers_states

    def _all_gather_module_weight(self, module):
        assert module not in self._bias_global_output and module not in self._weight_global_output
        with_bias = module.bias is not None

        # submit the all-gather communication for weight and bias.
        if with_bias:
            if module not in self._bias_global_output:
                bias_output, bias_handle = all_gather_raw(
                    module.bias,
                    self.process_group,
                    async_op=True,
                )
                self._bias_global_handle[module] = bias_handle
                self._bias_global_output[module] = bias_output

        if module not in self._weight_global_output:
            weight_output, weight_handle = all_gather_raw(
                module.weight,
                self.process_group,
                async_op=True,
            )
            self._weight_global_handle[module] = weight_handle
            self._weight_global_output[module] = weight_output

    def _all_gather_block_weight(self, block_index: int):
        block = self._index_to_block[block_index]

        # wait for prerequisite conditions
        if self.is_forward:
            for callback in self._forward_prefetch_prerequisites:
                callback(block)

        # prefetch parameters for all isp modules of the block
        for module in self._index_to_isp_modules[block_index]:
            self._all_gather_module_weight(module)

    def _wait_handle(self, module):
        handle = self._weight_global_handle[module]
        if handle is not None:
            handle.wait()

        if module.bias is None:
            return

        bias_handle = self._bias_global_handle[module]
        if bias_handle is not None:
            bias_handle.wait()

    def _clear_handle(self, module):
        if module in self._weight_global_handle:
            del self._weight_global_handle[module]
        if module in self._bias_global_handle:
            del self._bias_global_handle[module]

    def _clear_weight(self, module):
        if module in self._weight_global_output:
            del self._weight_global_output[module]
        if module in self._bias_global_output:
            del self._bias_global_output[module]

    def _pre_forward_hook_for_first_block(self, *args):  # pylint: disable=W0613
        """
        prefetch weight for block 0 before forward.
        """
        if self._forward_overlap_per == "layer" and self.is_forward is True:
            self._all_gather_block_weight(0)

    def _pre_forward_hook_for_prefetch_launch_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        block_index = self._module_to_index[module]

        if self._forward_overlap_per == "layer":
            # recompute
            if self.is_forward is False:
                if (block_index + 1) % self._ckpt_per_block_num == 0:
                    next_block_index = block_index - self._ckpt_per_block_num * 2 + 1
                else:
                    next_block_index = block_index + 1
                if 0 <= next_block_index < self._num_blocks:
                    self._all_gather_block_weight(next_block_index)
            else:
                # start the all-gather for next block
                if block_index + 1 < self._num_blocks:
                    self._all_gather_block_weight(block_index + 1)

    def _pre_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        if module not in self._weight_global_handle:
            self._all_gather_module_weight(module)

        self._wait_handle(module)

        if self._forward_overlap_per == "module":
            # start the all-gather for next module
            # 1.forward prefetch for next module
            module_index = self._isp_modules.index(module)
            module_layer_id = self._module_to_index[module]
            if module_index + 1 < len(self._isp_modules) and self.is_forward is True:
                next_module = self._isp_modules[module_index + 1]
                self._all_gather_module_weight(next_module)
            # 2.recompute forward prefetch for next module
            if self.is_forward is False:
                if module_index + 1 < len(self._isp_modules):
                    next_module = self._isp_modules[module_index + 1]
                    next_module_layer_id = self._module_to_index[next_module]
                    if module_layer_id == next_module_layer_id:
                        self._all_gather_module_weight(next_module)
                    # if current module is the last module in current layer, prefetch previous layer's first module
                    elif module_layer_id - 1 >= 0:
                        next_module = self._index_to_isp_modules[module_layer_id - 1][0]
                        self._all_gather_module_weight(next_module)
                else:
                    # if current module is the last module, prefetch previous layer's first module
                    if module_layer_id - 1 >= 0:
                        next_module = self._index_to_isp_modules[module_layer_id - 1][0]
                        self._all_gather_module_weight(next_module)

    def _post_forward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        if not ((self._module_to_index[module] < self._ckpt_block_num) and self.is_forward is False):
            self._clear_weight(module)
            self._clear_handle(module)

    def _pre_backward_hook_for_module(self, module: nn.Module, *args):  # pylint: disable=W0613
        # wait handle for current module
        if module not in self._weight_global_handle:
            self._all_gather_module_weight(module)

        self._wait_handle(module)

        # start the all-gather for next module
        module_index = self._isp_modules.index(module)
        if module_index - 1 >= 0:
            next_module = self._isp_modules[module_index - 1]
            if self._module_to_index[next_module] >= self._ckpt_block_num:
                self._all_gather_module_weight(next_module)

    def _post_backward_hook_for_module(self, module, *args):  # pylint: disable=W0613
        self._clear_handle(module)
        self._clear_weight(module)

    def _early_reduce_scatter_release_hook(self, *args):  # pylint: disable=W0613
        for handle in self._early_prev_layer_rs_handles:
            handle.wait()

        self._early_prev_layer_rs_handles = self._early_curr_layer_rs_handles
        self._early_curr_layer_rs_handles = []

    def _register_sync_parameters_hook(self) -> None:
        """
        register forward hooks and backward hooks for isp modules.
        """
        # register forward hooks
        # 1. register pre_forward_hook @block_0 to prefetch weight for block 0.
        # 2. register pre_forward_hook @prefetch_launch_module to prefetch weight for next block,
        #    when forward overlap granularity is 'layer'.
        # 3. register pre_forward_hook @isp_module to wait handle for current module,
        #    and prefetch weight for next module when forward overlap granularity is 'module'.
        # 4. register post_forward_hook @isp_module to release memory resource.
        self._index_to_block[0].register_forward_pre_hook(self._pre_forward_hook_for_first_block)

        for module in self._isp_prefetch_launch_module:
            module.register_forward_pre_hook(self._pre_forward_hook_for_prefetch_launch_module)

        for module in self._isp_modules:
            module.register_forward_pre_hook(self._pre_forward_hook_for_module)
            module.register_forward_hook(self._post_forward_hook_for_module)

        # register backward hooks
        # 1. register pre_backward_hook @isp_module to wait handle for current module and to prefetch for next module.
        # 2. register post_backward_hook @isp_module to release memory resource.
        if self._ckpt_block_num < self._num_blocks:
            for module in self._isp_modules:
                module.register_full_backward_pre_hook(self._pre_backward_hook_for_module)

        for module in self._isp_modules:
            module.register_full_backward_hook(self._post_backward_hook_for_module)

        if self._enable_early_reduce_scatter_release:
            for block_idx in range(self._num_blocks):
                block = self._index_to_block[block_idx]
                block.register_full_backward_hook(self._early_reduce_scatter_release_hook)

    def _get_constant_zero(self, size: tuple) -> torch.Tensor:
        if size not in self._zero_const_pool:
            self._zero_const_pool[size] = torch.zeros(
                *size, dtype=self.model_conf.dtype, device=self.model_conf.device
            ).contiguous()

        return self._zero_const_pool[size]

    def switch_current_model_chunk(self, chunk_id: int, is_llm: bool = True) -> None:
        if self.model_conf.pure_llm is False:
            self._mlp_isp_no_overlap = self._cid_to_mlp_isp_no_overlap[chunk_id]

        if is_llm:
            _overlap_states = self._overlap_states[chunk_id]["llm"]
        else:
            _overlap_states = self._overlap_states[chunk_id]["vision"]

        self._isp_prefetch_launch_module = _overlap_states.isp_prefetch_launch_module
        self._isp_modules = _overlap_states.isp_modules
        self._weight_global_handle = _overlap_states.weight_global_handle
        self._bias_global_handle = _overlap_states.bias_global_handle
        self._weight_global_output = _overlap_states.weight_global_output
        self._bias_global_output = _overlap_states.bias_global_output
        self._module_to_index = _overlap_states.module_to_index
        self._index_to_isp_modules = _overlap_states.index_to_isp_modules
        self._index_to_block = _overlap_states.index_to_block
        self._ckpt_block_num = _overlap_states.ckpt_block_num
        self._ckpt_per_block_num = _overlap_states.ckpt_per_block_num
        self._num_blocks = _overlap_states.num_blocks

    def register_prerequisite_for_forward_prefetch_hooks(self, prerequisite_func: Callable) -> None:
        """
        Registers a callback function that specifies a prerequisite condition for
        prefetching parameters before forward computation.

        This method allows users to define custom logic that must be satisfied before
        parameters are fetched for the next forward pass. This can be useful for
        implementing complex parameter update strategies or for coordinating
        parameter access with other system components.

        Args:
            prerequisite_func (Callable): A callable that represents the prerequisite
                                    condition. This function will be invoked before
                                    the parameters are prefetched, and its return value
                                    will determine whether the prefetching should proceed.

        Returns:
            None: This method does not return any value.

        Raises:
            TypeError: If the provided 'prerequisite_func' is not callable.
        """
        if not callable(prerequisite_func):
            raise TypeError("The provided prerequisite function must be callable.")

        self._forward_prefetch_prerequisites.append(prerequisite_func)

    def communication_mode(self) -> str:
        return "wp"

    def weight_hook(
        self, tensor: torch.Tensor, async_op: bool = False, module: nn.Module = None, is_bias: bool = False
    ) -> torch.Tensor:
        if dist.get_world_size(self.process_group) <= 1:
            return tensor

        if not self.overlap or module in self._mlp_isp_no_overlap:
            result, _ = all_gather_raw(tensor, self.process_group, async_op=async_op)
        elif is_bias:
            assert module is not None, "The module parameter must be specified"
            result = self._bias_global_output[module]
        else:
            assert module is not None, "The module parameter must be specified"
            result = self._weight_global_output[module]

        return result

    def grad_hook(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
        module: nn.Module = None,
        reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
        is_bias: bool = False,
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        if dist.get_world_size(self.process_group) <= 1:
            return tensor, DUMMY_HANDLE_CONST

        if not self.overlap or module in self._mlp_isp_no_overlap:
            result, handle = reduce_scatter_raw(tensor, self.process_group, op=reduce_op, async_op=async_op)
        else:
            assert module is not None, "The module parameter must be specified"

            if is_bias:
                assert hasattr(module.bias, "isp_reduce_scatter_name")
                key = getattr(module.bias, "isp_reduce_scatter_name")
            else:
                assert hasattr(module.weight, "isp_reduce_scatter_name")
                key = getattr(module.weight, "isp_reduce_scatter_name")

            output, handle = reduce_scatter_raw(
                tensor,
                self.process_group,
                op=reduce_op,
                async_op=async_op,
            )

            if self._enable_early_reduce_scatter_release:
                self._early_curr_layer_rs_handles.append(handle)

            self.reduce_scatter_handlers[key] = (output, handle)

            result, handle = (
                self._get_constant_zero(
                    (
                        tensor.shape[0] // dist.get_world_size(self.process_group),
                        *tensor.shape[1:],
                    )
                ),
                DUMMY_HANDLE_CONST,
            )

        return result, handle


class ISPCommunicatorSchedulerHook(SchedulerHook):
    """
    SchedulerHook for isp overlap handler
    """

    def __init__(self, overlap_handler: ISPCommunicator, zero_optim) -> None:
        self._isp_communicator = overlap_handler
        self._zero_optim = zero_optim

    def before_forward(self, scheduler, inputs) -> None:  # pylint: disable=W0613
        self._isp_communicator.is_forward = True
        # switch model chunk before forward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        _is_llm = False
        if self._isp_communicator.model_conf.pure_llm or gpc.get_local_rank(ParallelMode.PIPELINE) >= 1:
            _is_llm = True
        self._isp_communicator.switch_current_model_chunk(chunk_id, is_llm=_is_llm)

    def after_forward(self, scheduler, outputs) -> None:  # pylint: disable=W0613
        pass

    def before_criterion(self, scheduler, outputs, label) -> None:  # pylint: disable=W0613
        pass

    def after_criterion(self, scheduler, loss) -> None:  # pylint: disable=W0613
        pass

    def before_backward(self, scheduler, outputs, outputs_grad) -> None:  # pylint: disable=W0613
        self._isp_communicator.is_forward = False
        # switch model chunk before backward
        chunk_id = 0 if gpc.virtual_pipeline_parallel_rank is None else gpc.virtual_pipeline_parallel_rank
        _is_llm = True
        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE) and gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
            _is_llm = False
        self._isp_communicator.switch_current_model_chunk(chunk_id, is_llm=_is_llm)

    def after_backward(self, scheduler, inputs_grad) -> None:  # pylint: disable=W0613
        # accumulate left gradients in last bucket after backward.
        if self._isp_communicator and self._isp_communicator.overlap:
            self._zero_optim.accumulate_left_grads_after_backward()

        if self._isp_communicator and self._isp_communicator._enable_early_reduce_scatter_release:
            self._isp_communicator._early_prev_layer_rs_handles = []
            self._isp_communicator._early_curr_layer_rs_handles = []

    def post_helper_func(self, scheduler, outputs, label) -> None:  # pylint: disable=W0613
        pass


class ISPCommunicatorWrapper:
    """
    Wrapper for multiple ISPCommunicators.
    TODO: check all isp communicator external interfaces and wrap them.
    """

    def __init__(
        self,
        isp_communicators: List[ISPCommunicator],
    ) -> None:
        self.isp_communicators = isp_communicators
        self.reduce_scatter_handlers = {}

        self.reduce_scatter_handlers = UniqueChainMap(
            *(isp_communicator.reduce_scatter_handlers for isp_communicator in self.isp_communicators)
        )

    def register_prerequisite_for_forward_prefetch_hooks(self, prerequisite_func: Callable) -> None:
        for isp_communicator in self.isp_communicators:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(prerequisite_func)


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class _SeqAllToAll(torch.autograd.Function):
    "sequence alltoall function"

    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        scatter_idx: Optional[Union[List[int], int]],
        gather_idx: Optional[Union[List[int], int]],
        *input_: torch.Tensor,
    ) -> torch.Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        seq_world_size = dist.get_world_size(group)

        if dist.get_world_size(group) <= 1:
            if len(input_) == 1:
                return input_[0]
            return input_

        if len(input_) == 1:
            input_list = [t.contiguous() for t in torch.tensor_split(input_[0], seq_world_size, scatter_idx)]
            output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
            # TODO: use all_to_all_single instead
            dist.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()

        outputs = []

        assert len(scatter_idx) == len(gather_idx)
        assert len(gather_idx) == len(input_)

        for i in range(len(input_)):

            if i == 0:
                input_list = [t.contiguous() for t in torch.tensor_split(input_[i], seq_world_size, scatter_idx[i])]
                output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
                handle_last = dist.all_to_all(output_list, input_list, group=group, async_op=True)

            # conduct the next all2all
            if i + 1 < len(input_):
                input_list_next = [
                    t.contiguous() for t in torch.tensor_split(input_[i + 1], seq_world_size, scatter_idx[i + 1])
                ]
                output_list_next = [torch.empty_like(input_list_next[0]) for _ in range(seq_world_size)]
                handle_next = dist.all_to_all(output_list_next, input_list_next, group=group, async_op=True)

            handle_last.wait()

            outputs.append(torch.cat(output_list, dim=gather_idx[i]).contiguous())

            if i + 1 < len(input_):
                handle_last = handle_next
                input_list = input_list_next
                output_list = output_list_next

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:

        if dist.get_world_size(ctx.group) <= 1:
            return (None, None, None, *grad_output)
        res = _SeqAllToAll.apply(ctx.group, ctx.gather_idx, ctx.scatter_idx, *grad_output)
        if len(grad_output) == 1:
            return (None, None, None, res)

        return (None, None, None, *res)


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class DistributedAttention(nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local self-attention module
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(
        self,
        local_attention: Union[nn.Module, Callable],
        sequence_process_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.sp_size = dist.get_world_size(self.spg)

    @params_dispatch_with_condition(condition=check_attention_argument)
    def forward(self) -> torch.Tensor:
        assert False, "Should never arrive"

    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.WithOut)))
    def _qkv(self, qkv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            qkv (Tensor): packed qkv input to the layer
            kwargs: other args

        Returns:
            * output (Tensor): context output
        """

        # qkv shape: [1, packlen, 3, n_head, head_dim] or [batch, seqlen, 3, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        qkv = _SeqAllToAll.apply(self.spg, 3, 1, qkv)

        context = self.local_attn(qkv, *args, **kwargs)

        # context shape: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in seqlen(packlen) and gather in n_head
        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.WithOut)))
    def _q_kv(self, q: torch.Tensor, kv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            q (Tensor): q input to the layer
            kv (Tensor): packed kv input to the layer
            kwargs: other args

        Returns:
            output (Tensor): context output
        """
        # q shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)

        # kv shape: [1, packlen, 2, n_head, head_dim] or [batch, seqlen, 2, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        num_head_kv = kv.shape[3]
        # if the num head of kv is not enough to be splitted by sp
        # then we could copy the kv head
        if self.sp_size > num_head_kv:
            assert self.sp_size % num_head_kv == 0, "the num_head_kv should be divided by sp size."
            kv = expandKVPacked(kv, self.sp_size // num_head_kv, 3)

        q, kv = _SeqAllToAll.apply(self.spg, [2, 3], [1, 1], q, kv)

        context = self.local_attn(q, kv, *args, **kwargs)

        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.With)))
    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.WithOut)))
    def _q_k_v(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """forward

        Arguments:
            q (Tensor): q input to the layer
            k (Tensor): k input to the layer
            v (Tensor): v input to the layer
            kwargs: other args

        Returns:
            * output (Tensor): context output
        """
        # self._scatter_gather_idx["q"] = [1, 0]  # q/k/v shape: [sequence, head, head_dim]
        # q shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        q = _SeqAllToAll.apply(self.spg, 2, 1, q)
        # k shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        k = _SeqAllToAll.apply(self.spg, 2, 1, k)
        # v shpae: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in n_head and gather in seqlen(packlen)
        v = _SeqAllToAll.apply(self.spg, 2, 1, v)

        context = self.local_attn(q, k, v, *args, **kwargs)

        # context shape: [1, packlen, n_head, head_dim] or [batch, seqlen, n_head, head_dim]
        # scatter in seqlen(packlen) and gather in n_head
        context = _SeqAllToAll.apply(self.spg, 1, 2, context)

        return context


def auto_wrap_distributed_attention(attn_impl: nn.Module) -> Callable[[bool, Any, float], nn.Module]:
    """
    Wrap a local attention module to a distributed one, which will be used in the ISP parallelism.
    """

    # should we impl distributed attention as a metaclass?
    def _attetion_constructor(
        attn_impl: type, causal=False, softmax_scale=None, attention_dropout=0.0, layer_idx=0
    ) -> nn.Module:
        tp_mode = gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name)

        if tp_mode != TensorParallelMode.isp.name:
            return attn_impl(causal, softmax_scale, attention_dropout)
        else:
            if gpc.config.parallel.sequence_2D.enable is True:
                spg = gpc.get_group(ParallelMode.HEAD)
            else:
                spg = gpc.get_group(ParallelMode.TENSOR)
            return DistributedAttention(
                local_attention=attn_impl(causal, softmax_scale, attention_dropout, layer_idx),
                sequence_process_group=spg,
            )

    return partial(_attetion_constructor, attn_impl=attn_impl)


def auto_wrap_func_distributed_attention(attn_impl: Callable) -> Callable[..., Callable]:
    """
    Wrap a local attention function to a distributed one, which will be used in the ISP parallelism.
    """

    # should we impl distributed attention as a metaclass?
    def _attetion_constructor(*args, attn_impl: type, **kwargs) -> Callable:
        tp_mode = gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name)

        if tp_mode != TensorParallelMode.isp.name:
            return attn_impl(*args, **kwargs)
        else:
            return DistributedAttention(
                local_attention=attn_impl, sequence_process_group=gpc.get_group(ParallelMode.TENSOR)
            )(*args, **kwargs)

    return partial(_attetion_constructor, attn_impl=attn_impl)
