from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from packaging.version import Version
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_neo_chat import NEOMoELLMConfig
from .modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    create_block_causal_mask,
)
from .transformers_compat import causal_mask_kwargs, model_input_compat, tied_weights_keys


def _torch_version() -> Version:
    return Version(torch.__version__.split("+", 1)[0])


def _cuda_grouped_mm_arch_supported(
    torch_version: Version,
    capability: tuple[int, int],
) -> bool:
    """Apply the architecture restrictions of the available PyTorch API."""

    if torch_version < Version("2.9"):
        return capability[0] == 9
    return capability[0] >= 8


def _grouped_mm_strides_supported(tensor: torch.Tensor) -> bool:
    """Mirror PyTorch's 16-byte row/column-major stride validation."""

    if tensor.dim() not in {2, 3}:
        return False
    row_stride, column_stride = tensor.stride()[-2:]
    rows, columns = tensor.shape[-2:]
    alignment = max(1, 16 // tensor.element_size())
    if row_stride == 1 and column_stride >= max(1, rows):
        return column_stride % alignment == 0
    if column_stride == 1 and row_stride >= max(1, columns):
        return row_stride % alignment == 0
    return False


def _grouped_mm_runtime_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Return whether this PyTorch/device combination has a safe grouped MM."""

    public_op = getattr(F, "grouped_mm", None)
    if device.type == "cpu":
        return (
            public_op is not None
            and _torch_version() >= Version("2.9")
            and dtype in {torch.float16, torch.bfloat16, torch.float32}
        )
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or dtype != torch.bfloat16
        or not (public_op is not None or hasattr(torch, "_grouped_mm"))
    ):
        return False
    try:
        capability = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError):
        return False
    return _cuda_grouped_mm_arch_supported(_torch_version(), capability)


def _grouped_mm_supported(input_: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether PyTorch's grouped MM is safe for these operands.

    PyTorch 2.8 only dispatches ``torch._grouped_mm`` on CUDA.  Later releases
    added CPU support and the public functional API.  CUDA's dense grouped
    kernel is deliberately limited to bf16 here: that is the reference
    inference dtype and avoids private-kernel dtype/architecture failures on
    older PyTorch releases.  Unsupported combinations transparently retain the
    eager expert implementation.
    """

    if input_.device != weight.device or input_.dtype != weight.dtype:
        return False
    if input_.layout != torch.strided or weight.layout != torch.strided:
        return False
    if input_.dim() != 2 or weight.dim() != 3:
        return False
    if not (
        _grouped_mm_strides_supported(input_)
        and _grouped_mm_strides_supported(weight)
    ):
        return False
    if not _grouped_mm_runtime_supported(input_.device, input_.dtype):
        return False

    if input_.device.type == "cpu":
        # Early CPU kernels also require 16-byte-aligned base pointers.  Packed
        # expert weights normally satisfy this, but sliced inputs need not.
        if _torch_version().release[:2] <= (2, 10):
            return input_.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0
        return True

    if any(dimension % 8 for dimension in (*input_.shape[1:], *weight.shape[1:])):
        return False
    return True


def _grouped_mm(input_: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Call the public grouped-MM API when present, then the PyTorch 2.8 API."""

    public_op = getattr(F, "grouped_mm", None)
    if public_op is not None:
        return public_op(input_, weight, offs=offsets)
    return torch._grouped_mm(input_, weight, offs=offsets)


class Qwen3MoeMLP(nn.Module):
    """Single expert FFN. Same structure as :class:`Qwen3MLP` but the
    intermediate size is parameterised so it can be ``moe_intermediate_size``
    (per-expert) for experts and ``intermediate_size`` for any dense fallback.
    """

    def __init__(self, config, intermediate_size: Optional[int] = None):
        super().__init__()
        from transformers.activations import ACT2FN

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeExperts(nn.ModuleList):
    """Checkpoint-compatible experts with optional packed inference weights.

    The public module hierarchy remains a ``ModuleList`` of
    ``gate_proj/up_proj/down_proj`` layers, so existing A3B checkpoints,
    LoRAs, conversion tools, and saved state dicts retain exactly the same
    keys.  For grouped inference, the dense parameters are lazily
    copied once into two contiguous storages and each public parameter is
    rebound to its corresponding view.  This avoids rebuilding a multi-GB
    stack on every decoder layer invocation without keeping a duplicate cache.

    Training always takes the eager path.  The packed tensors are intentionally
    inference-only aliases of the registered per-expert Parameters, so using
    them while gradients are enabled would otherwise bypass expert gradients.
    """

    def __init__(
        self,
        config: NEOMoELLMConfig,
        num_experts: int,
        intermediate_size: int,
    ):
        experts = [
            Qwen3MoeMLP(config, intermediate_size=intermediate_size)
            for _ in range(num_experts)
        ]
        super().__init__(experts)
        self.config = config
        self.num_experts = int(num_experts)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(intermediate_size)

        # Plain tensor attributes, rather than buffers/parameters: the public
        # state dict must remain in the original per-expert checkpoint layout.
        object.__setattr__(self, "_packed_gate_up_proj", None)
        object.__setattr__(self, "_packed_down_proj", None)
        self._last_experts_implementation = "eager"
        self.register_load_state_dict_post_hook(self._clear_stale_packed_weights_after_load)

    def __getitem__(self, index):
        # ModuleList's slicing constructor assumes a single iterable argument.
        if isinstance(index, slice):
            return nn.ModuleList(list(self)[index])
        return super().__getitem__(index)

    def _clear_stale_packed_weights_after_load(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        self.release_stale_packed_weights()

    def release_stale_packed_weights(self) -> None:
        """Drop packed storage after an external parameter-data rebind.

        Accelerate, layer offload, ``load_state_dict(assign=True)``, and a few
        quantizers replace ``Parameter.data`` without going through this
        module.  Retaining the old packed tensor in that case would both use
        stale values and, for CPU offload, keep the old GPU allocation alive.
        """

        if not self._weights_are_packed():
            object.__setattr__(self, "_packed_gate_up_proj", None)
            object.__setattr__(self, "_packed_down_proj", None)

    def _apply(self, fn, recurse=True):
        if (
            recurse
            and not self.training
            and self._weights_are_packed()
            and all(parameter.grad is None for parameter in self.parameters())
            and not torch.__future__.get_overwrite_module_params_on_conversion()
            and not torch.__future__.get_swap_module_params_on_conversion()
        ):
            # Moving the two backing tensors once is materially cheaper than
            # invoking .to() for every expert slice, and it preserves the
            # packed layout across the common CPU-load -> GPU-inference path.
            gate_up = fn(self._packed_gate_up_proj)
            down = fn(self._packed_down_proj)
            if gate_up.device.type == "meta" or down.device.type == "meta":
                result = super()._apply(fn, recurse=recurse)
                self.release_stale_packed_weights()
                return result
            self._bind_packed_weights(gate_up, down)

            # Expert activations currently have no state, but recurse through
            # any non-projection child to remain correct if that changes.
            for expert in self:
                for name, child in expert.named_children():
                    if name not in {"gate_proj", "up_proj", "down_proj"}:
                        child._apply(fn)
            return self

        result = super()._apply(fn, recurse=recurse)
        self.release_stale_packed_weights()
        return result

    def _bind_packed_weights(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        for expert_idx, expert in enumerate(self):
            expert.gate_proj.weight.data = gate_up[expert_idx, : self.intermediate_size]
            expert.up_proj.weight.data = gate_up[expert_idx, self.intermediate_size :]
        object.__setattr__(self, "_packed_gate_up_proj", gate_up)

        for expert_idx, expert in enumerate(self):
            expert.down_proj.weight.data = down[expert_idx]
        object.__setattr__(self, "_packed_down_proj", down)

    @staticmethod
    def _shares_storage(tensor: torch.Tensor, packed: torch.Tensor, offset: int) -> bool:
        if tensor.device.type == "meta" or packed.device.type == "meta":
            return False
        return (
            tensor.device == packed.device
            and tensor.dtype == packed.dtype
            and tensor.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
            and tensor.storage_offset() == offset
            and tensor.is_contiguous()
        )

    def _weights_are_packed(self) -> bool:
        gate_up = self._packed_gate_up_proj
        down = self._packed_down_proj
        if gate_up is None or down is None:
            return False
        if gate_up.shape != (
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
        ):
            return False
        if down.shape != (
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        ):
            return False

        gate_stride = 2 * self.intermediate_size * self.hidden_size
        down_stride = self.hidden_size * self.intermediate_size
        up_offset = self.intermediate_size * self.hidden_size
        for expert_idx, expert in enumerate(self):
            if any(
                type(getattr(expert, projection_name, None)) is not nn.Linear
                or getattr(expert, projection_name).bias is not None
                for projection_name in ("gate_proj", "up_proj", "down_proj")
            ):
                # A wrapper/subclass may add LoRA deltas or otherwise override
                # Linear.forward; the packed path must never bypass it.
                return False
            if not self._shares_storage(
                expert.gate_proj.weight,
                gate_up,
                expert_idx * gate_stride,
            ):
                return False
            if not self._shares_storage(
                expert.up_proj.weight,
                gate_up,
                expert_idx * gate_stride + up_offset,
            ):
                return False
            if not self._shares_storage(
                expert.down_proj.weight,
                down,
                expert_idx * down_stride,
            ):
                return False
        return True

    @torch.no_grad()
    @torch.inference_mode(False)
    def _pack_weights(self) -> bool:
        """Coalesce dense expert weights and preserve all Parameter objects."""

        # A replacement may require eager execution and invalidate the cache
        # at the same time. Release its old storage before any early return.
        self.release_stale_packed_weights()
        # Calling raw GEMMs bypasses Module.__call__/forward on every expert
        # and projection. Hooks (including Accelerate placement hooks) and
        # custom forwards must keep their original execution semantics.
        if not self._plain_experts_supported():
            return False
        if self._packed_gate_up_proj is not None:
            return True
        if not self:
            return False

        expected = (
            ("gate_proj", (self.intermediate_size, self.hidden_size)),
            ("up_proj", (self.intermediate_size, self.hidden_size)),
            ("down_proj", (self.hidden_size, self.intermediate_size)),
        )
        first_weight = getattr(getattr(self[0], "gate_proj", None), "weight", None)
        if (
            not isinstance(first_weight, nn.Parameter)
            or first_weight.device.type == "meta"
            or first_weight.layout != torch.strided
            or not first_weight.dtype.is_floating_point
        ):
            return False

        for expert in self:
            for projection_name, shape in expected:
                projection = getattr(expert, projection_name, None)
                weight = getattr(projection, "weight", None)
                if (
                    type(projection) is not nn.Linear
                    or not isinstance(weight, nn.Parameter)
                    or projection.bias is not None
                    or tuple(weight.shape) != shape
                    or weight.device != first_weight.device
                    or weight.dtype != first_weight.dtype
                    or weight.layout != torch.strided
                ):
                    # Quantized/GGUF experts retain their own eager operators.
                    return False

        if first_weight.is_cuda:
            # Offload may allocate on a prefetch stream. Rebinding .data below
            # frees those storages before the layer's post-hook can protect
            # them, so keep them alive until the packing copies finish.
            stream = torch.cuda.current_stream(first_weight.device)
            for expert in self:
                for projection_name, _ in expected:
                    getattr(expert, projection_name).weight.record_stream(stream)

        gate_up = first_weight.new_empty(
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
        )
        for expert_idx, expert in enumerate(self):
            gate_up[expert_idx, : self.intermediate_size].copy_(expert.gate_proj.weight)
            gate_up[expert_idx, self.intermediate_size :].copy_(expert.up_proj.weight)

        # Release the old gate/up storages before allocating the packed down
        # projection.  This keeps lazy packing's transient peak at roughly
        # 1.67x one expert block rather than 2x.
        for expert_idx, expert in enumerate(self):
            expert.gate_proj.weight.data = gate_up[expert_idx, : self.intermediate_size]
            expert.up_proj.weight.data = gate_up[expert_idx, self.intermediate_size :]
        object.__setattr__(self, "_packed_gate_up_proj", gate_up)

        down = first_weight.new_empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        for expert_idx, expert in enumerate(self):
            down[expert_idx].copy_(expert.down_proj.weight)

        # Rebinding .data (rather than replacing Parameters) preserves hooks,
        # optimizer references, Accelerate/offload bookkeeping, and public keys.
        for expert_idx, expert in enumerate(self):
            expert.down_proj.weight.data = down[expert_idx]
        object.__setattr__(self, "_packed_down_proj", down)
        return True

    def _plain_experts_supported(self) -> bool:
        module_hooks = nn.modules.module
        if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
            return False
        for expert in self:
            if type(expert) is not Qwen3MoeMLP:
                return False
            for child in expert.modules():
                if (
                    child._forward_hooks
                    or child._forward_pre_hooks
                    or "forward" in child.__dict__
                    or hasattr(child, "_hf_hook")
                ):
                    return False
        return True

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states.index_select(0, token_idx)
            current_out = expert_layer(current_state) * routing_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, current_out.to(hidden_states.dtype))
        return output

    def _grouped_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Reject mixed device/dtype placement before allocating packed weights.
        if not self:
            return None
        first_weight = getattr(getattr(self[0], "gate_proj", None), "weight", None)
        if not isinstance(first_weight, torch.Tensor) or (
            first_weight.device != hidden_states.device
            or first_weight.dtype != hidden_states.dtype
        ):
            self.release_stale_packed_weights()
            return None
        alignment = max(1, 16 // hidden_states.element_size())
        if (
            not _grouped_mm_runtime_supported(hidden_states.device, hidden_states.dtype)
            or not _grouped_mm_strides_supported(hidden_states)
            or self.hidden_size % alignment
            or self.intermediate_size % alignment
        ):
            self.release_stale_packed_weights()
            return None
        if not self._pack_weights():
            return None

        gate_up_weight = self._packed_gate_up_proj.transpose(1, 2)
        down_weight = self._packed_down_proj.transpose(1, 2)
        # Validate both GEMMs before launching either one.  In particular,
        # 2 * intermediate_size can meet CUDA's multiple-of-eight constraint
        # while intermediate_size itself does not.
        down_input_probe = hidden_states.new_empty((0, self.intermediate_size))
        if not (
            _grouped_mm_supported(hidden_states, gate_up_weight)
            and _grouped_mm_supported(down_input_probe, down_weight)
        ):
            return None

        num_tokens = hidden_states.shape[0]
        top_k = selected_experts.shape[1]
        expert_ids = selected_experts.reshape(-1)
        expert_ids_grouped, permutation = torch.sort(expert_ids)
        grouped_states = hidden_states[permutation // top_k]
        grouped_routing_weights = routing_weights.reshape(-1)[permutation]

        # histc has a fixed-size result and remains CUDA-graph friendly.  CPU
        # histc requires floating input; CUDA accepts the cheaper int32 form.
        histc_input = (
            expert_ids_grouped.float()
            if hidden_states.device.type == "cpu"
            else expert_ids_grouped.int()
        )
        tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up = _grouped_mm(grouped_states, gate_up_weight, offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self[0].act_fn(gate) * up
        projected = _grouped_mm(intermediate, down_weight, offsets)
        weighted = projected * grouped_routing_weights.unsqueeze(-1)

        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(),
            device=permutation.device,
        )
        weighted = weighted[inverse_permutation]
        return weighted.reshape(num_tokens, top_k, self.hidden_size).sum(dim=1).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        implementation = getattr(self.config, "_experts_implementation", None)
        if implementation is None:
            implementation = getattr(self.config, "experts_implementation", None)
        implementation = implementation or "eager"
        if implementation not in {"eager", "grouped_mm"}:
            raise ValueError(
                f"Unsupported experts implementation {implementation!r}; expected "
                "'eager' or 'grouped_mm'."
            )

        # Packed aliases are not registered Parameters.  Keep every autograd
        # use case on the reference path, including frozen-base fine-tuning
        # where only input/router/adapter gradients may be required.
        # Reentrant gradient checkpointing runs its first training forward
        # under no_grad. It must use the same arithmetic as the recomputation.
        if (
            implementation == "eager"
            or self.training
            or torch.is_grad_enabled()
            # grouped_mm does not follow Linear's autocast policy.
            or torch.is_autocast_enabled(hidden_states.device.type)
        ):
            self._last_experts_implementation = "eager"
            return self._eager_forward(hidden_states, selected_experts, routing_weights)

        optimized_output = None
        if (
            implementation == "grouped_mm"
            and hidden_states.numel() > 0
            and selected_experts.numel() > 0
        ):
            optimized_output = self._grouped_forward(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        if optimized_output is None:
            self._last_experts_implementation = "eager"
            return self._eager_forward(hidden_states, selected_experts, routing_weights)

        self._last_experts_implementation = implementation
        return optimized_output


@torch.no_grad()
def prepare_moe_experts_for_inference(module: nn.Module, device=None) -> int:
    """Pack dense MoE experts before moving a normally loaded model to GPU.

    Returns the number of expert collections packed.  Loading utilities call
    this while the model is still on CPU; :meth:`Qwen3MoeExperts._apply` then
    transfers each packed backing tensor once and avoids a transient duplicate
    of a full GPU expert block on the first forward.  Unsupported quantized or
    heterogeneous expert modules are simply left on their eager path.
    """

    packed = 0
    for submodule in module.modules():
        if not isinstance(submodule, Qwen3MoeExperts):
            continue
        implementation = getattr(submodule.config, "_experts_implementation", None)
        if implementation is None:
            implementation = getattr(submodule.config, "experts_implementation", None)
        if implementation != "grouped_mm" or submodule.training:
            continue
        if device is not None:
            weight = next(submodule.parameters())
            if not _grouped_mm_runtime_supported(torch.device(device), weight.dtype):
                continue
        packed += int(submodule._pack_weights())
    return packed


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k softmax-routed MoE block matching HuggingFace's Qwen3-MoE layout.

    Parameter names (``gate.weight``, ``experts.{i}.gate_proj/up_proj/down_proj``)
    are kept identical so converted A3B checkpoints load directly via the
    ``mlp.*`` / ``mlp_mot_gen.*`` keys. The block is parameterised explicitly
    so the same class can serve both the understanding branch (``num_experts``
    experts, top-k = ``num_experts_per_tok``, width ``moe_intermediate_size``)
    and the image-generation branch (``gen_num_experts`` etc.).
    """

    def __init__(
        self,
        config: NEOMoELLMConfig,
        num_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = int(num_experts) if num_experts is not None else int(config.num_experts)
        self.top_k = int(
            num_experts_per_tok if num_experts_per_tok is not None else config.num_experts_per_tok
        )
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.hidden_size = config.hidden_size

        expert_intermediate_size = int(
            moe_intermediate_size
            if moe_intermediate_size is not None
            else config.moe_intermediate_size
        )

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = Qwen3MoeExperts(
            config,
            num_experts=self.num_experts,
            intermediate_size=expert_intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        output = self.experts(flat, selected_experts, routing_weights)
        return output.view(*orig_shape)


class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    """A Qwen3-MoE decoder block with the NEO-Unify two-branch structure.

    Mirrors ``Qwen3DecoderLayer`` from :mod:`modeling_qwen3` but uses sparse
    MoE blocks on *both* branches:

      * ``self.mlp``         - understanding-path MoE
                               (``num_experts`` / ``num_experts_per_tok`` /
                               ``moe_intermediate_size``)
      * ``self.mlp_mot_gen`` - image-generation-path MoE
                               (``gen_num_experts`` / ``gen_num_experts_per_tok`` /
                               ``gen_moe_intermediate_size``)

    Layers listed in ``mlp_only_layers`` or those not aligned with
    ``decoder_sparse_step`` fall back to a dense :class:`Qwen3MoeMLP` on the
    understanding branch (matching upstream Qwen3-MoE), while the
    generation branch still uses a sparse MoE.
    """

    def __init__(self, config: NEOMoELLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        mlp_only_layers = list(getattr(config, "mlp_only_layers", []) or [])
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
        is_sparse = (
            int(config.num_experts) > 0
            and layer_idx not in mlp_only_layers
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(
                config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        # Image-generation branch: in the A3B checkpoint this is *also* a sparse
        # MoE block (``gen_num_experts`` experts, typically smaller than the und
        # branch's ``num_experts``). ``NEOMoELLMConfig`` defaults the gen-path
        # knobs to their und-path counterparts so legacy single-pool configs
        # keep working.
        self.mlp_mot_gen = Qwen3MoeSparseMoeBlock(
            config,
            num_experts=getattr(config, "gen_num_experts", config.num_experts),
            num_experts_per_tok=getattr(
                config, "gen_num_experts_per_tok", config.num_experts_per_tok
            ),
            moe_intermediate_size=getattr(
                config, "gen_moe_intermediate_size", config.moe_intermediate_size
            ),
        )

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states, image_gen_indicators, exist_non_image_gen_tokens,
                exist_image_gen_tokens, indexes, attention_mask, position_ids,
                past_key_values, use_cache, cache_position, **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states, image_gen_indicators, exist_non_image_gen_tokens,
                exist_image_gen_tokens, indexes, attention_mask, position_ids,
                past_key_values, use_cache, cache_position, **kwargs,
            )

        # Mixed und/gen path — see the NOTE in Qwen3Attention.forward (modeling_qwen3.py).
        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        # Mixed batch: dispatch tokens per branch then merge back. Matches the
        # dense ``Qwen3DecoderLayer.forward`` mixed-path implementation.
        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.input_layernorm(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.input_layernorm_mot_gen(
                hidden_states[image_gen_indicators]
            )
        hidden_states = _hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            und_hidden = self.post_attention_layernorm(
                hidden_states[~image_gen_indicators]
            )
            # MoE expects a 3D input (batch, seq, hidden); promote then squeeze.
            if und_hidden.dim() == 2:
                und_hidden = und_hidden.unsqueeze(0)
                _hidden_states[~image_gen_indicators] = self.mlp(und_hidden).squeeze(0)
            else:
                _hidden_states[~image_gen_indicators] = self.mlp(und_hidden)
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.mlp_mot_gen(
                self.post_attention_layernorm_mot_gen(hidden_states[image_gen_indicators])
            )

        hidden_states = _hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoePreTrainedModel(PreTrainedModel):
    config: NEOMoELLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False  # MoE routing has data-dependent control flow.
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3Attention,
    }

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Declare the local eager/grouped dispatch to Transformers 5."""

        return True

    def get_correct_experts_implementation(self, requested_experts=None):
        # Hardware eligibility is checked locally on the actual operands.
        # Transformers' global checks can reject a supported eager fallback.
        implementation = requested_experts or "eager"
        if implementation not in {"eager", "grouped_mm"}:
            raise ValueError("experts_implementation must be 'eager' or 'grouped_mm'")
        return implementation


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: NEOMoELLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_gen_indicators: Optional[torch.Tensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Convert the CUDA reductions once before the decoder loop. If the
            # scalar tensors reach every layer, each Python branch can force a
            # host-device synchronization and collapse async weight prefetch.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(input_ids.device)
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            self.current_index = indexes[0].max()

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        elif not exist_non_image_gen_tokens:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(hidden_states[~image_gen_indicators])
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(hidden_states[image_gen_indicators])
            hidden_states = _hidden_states

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys("lm_head.weight", "model.embed_tokens.weight")
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: NEOMoELLMConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeExperts",
    "Qwen3MoeMLP",
    "prepare_moe_experts_for_inference",
]
