# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""LoRA support for the SenseNova-U1 flow-matching (image-generation) path.

Side-branch design:

    y = base(x) + scale * lora_B(lora_A(dropout(x)))         where scale = alpha / r

The base linear is left untouched, including its parallel-comm wrapping
(ColumnParallelLinear under ISP), so its forward/backward communication is
unchanged. The LoRA branch is two plain ``nn.Linear`` modules replicated
across WP/TP ranks; under ISP the base layer consumes and produces full-dim
activations (weights are all-gathered inside forward), so the replicated
branch computes the same delta on every rank. The branch params fall under
``fm_modules.*`` and are therefore tagged ``IS_REPLICA_ZERO_PARALLEL`` by
``set_parallel_attr_for_param_groups`` — init sync and grad reduction reuse
the existing replica machinery.

We only target ``fm_modules.*`` (timestep_embedder, fm_head,
vision_model_mot_gen, optional noise_scale_embedder) by default.
"""
from __future__ import annotations

import fnmatch
import logging
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.shard import get_tensor_split_parallel_mode
from sensenovalm.model.modules.linear import (
    ColumnParallelLinear,
    ParallelLinearWithCommExt,
    RowParallelLinear,
)
from sensenovalm.utils.parallel import is_using_isp


logger = logging.getLogger(__name__)


def _base_full_features(base: nn.Linear) -> Tuple[int, int]:
    """Runtime (in, out) feature sizes of ``base``.

    For a parallel linear, ``in_features``/``out_features`` are the *per-rank
    shard* sizes, while under ISP the layer's runtime input/output are
    full-dim (the sharded weight is all-gathered inside forward). The LoRA
    branch must be built with the full dimensions.
    """
    if not isinstance(base, ParallelLinearWithCommExt):
        return base.in_features, base.out_features

    world_size = gpc.get_world_size(get_tensor_split_parallel_mode())
    if world_size == 1:
        return base.in_features, base.out_features
    if not is_using_isp():
        raise NotImplementedError(
            "LoRA on tensor-parallel (mtp/msp/fsp) linears is unsupported: their "
            "activations are sharded along the hidden dim, so a replicated LoRA "
            "branch would compute a wrong delta. Use tp_size=1 or the isp mode."
        )
    if isinstance(base, ColumnParallelLinear):
        return base.in_features, base.out_features * world_size
    if isinstance(base, RowParallelLinear):
        return base.in_features * world_size, base.out_features
    raise NotImplementedError(f"Cannot derive full feature sizes for {type(base).__name__}.")


class LoRALinear(nn.Module):
    """Side-branch LoRA adapter wrapping an existing ``nn.Linear``-like layer.

    ``lora_B`` is zero-initialized, so the adapter starts as a no-op.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        in_features, out_features = _base_full_features(base)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scale={self.scale:.3f}"

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Parallel-linear forwards may take extra args (batch_sizes, ...) and
        # may return a tuple; the LoRA delta applies to the primary output.
        base_out = self.base(x, *args, **kwargs)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        if isinstance(base_out, tuple):
            primary, *rest = base_out
            return (primary + delta, *rest)
        return base_out + delta

    # Mirror nn.Linear's surface for code that introspects the wrapped layer
    # (e.g. TimestepEmbedder reads ``self.mlp[0].weight.dtype``).
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def _set_submodule(root: nn.Module, qualname: str, new_module: nn.Module) -> None:
    parent_path, _, leaf = qualname.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new_module)


def inject_lora(
    model: nn.Module,
    target_prefixes: Sequence[str] = ("fm_modules.",),
    target_leaf_names: Sequence[str] = (
        # Vision-MoT attention / MLP (NEOVisionModel)
        "qkv",
        "proj",
        "fc1",
        "fc2",
        # Standard projection names, in case a future config rewires
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    include_sequential_indices: bool = True,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    skip_substrings: Sequence[str] = ("lora_A", "lora_B"),
) -> List[str]:
    """Inject LoRA adapters into every nn.Linear under ``target_prefixes``.

    Freezes the entire model first, then wraps the matching linears; the new
    LoRA params are the only trainable ones afterwards.

    Args:
        model: the SenseNova-U1 model (after build + pretrained weight loading).
        target_prefixes: only wrap layers whose dotted qualname starts with one
            of these.
        target_leaf_names: glob patterns matched against the layer's *leaf*
            attribute name. Use ``("*",)`` to match all.
        include_sequential_indices: also wrap numeric leaf names like ``"0"`` —
            the Linear children of ``TimestepEmbedder.mlp`` and ``fm_head``
            are addressed this way.
        r / alpha / dropout: LoRA hyper-parameters (effective scale = alpha / r).
        skip_substrings: never wrap a module whose qualname contains one of
            these (guards against double-wrapping on a second call).

    Returns:
        The list of wrapped module qualnames.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got r={r}")

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue  # keep adapters from a previous call trainable
        p.requires_grad = False

    # Collect the targets first to avoid mutating the module tree mid-walk.
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        if not any(name.startswith(p) for p in target_prefixes):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if _matches_any(leaf, target_leaf_names) or (include_sequential_indices and leaf.isdigit()):
            candidates.append((name, module))

    if not candidates:
        logger.warning(
            "inject_lora: no nn.Linear matched under prefixes=%s with leaf_names=%s. "
            "LoRA training will have zero trainable params!",
            list(target_prefixes),
            list(target_leaf_names),
        )

    wrapped_names: List[str] = []
    for name, linear in candidates:
        _set_submodule(model, name, LoRALinear(linear, r=r, alpha=alpha, dropout=dropout))
        wrapped_names.append(name)

    return wrapped_names


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA parameters (lora_A / lora_B) from the model."""
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def load_lora_state_dict(model: nn.Module, state: dict, strict: bool = True) -> None:
    """Load a LoRA-only state dict produced by :func:`lora_state_dict`."""
    own = {k for k in model.state_dict() if ".lora_A." in k or ".lora_B." in k}
    missing = sorted(own - set(state))
    unexpected = sorted(set(state) - own)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"load_lora_state_dict mismatch: missing={missing[:8]}{'...' if len(missing) > 8 else ''}, "
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    # strict=False because ``state`` intentionally omits the (frozen) base weights.
    model.load_state_dict(state, strict=False)


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total
