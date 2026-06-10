# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""LoRA support for the SenseNova-U1 flow-matching (image-generation) path.

Design choice — side-branch LoRA:

    y = base(x) + scale * lora_B(lora_A(dropout(x)))         where scale = alpha / r

The base ``nn.Linear`` is left untouched, including its parallel-comm wrapping
(ColumnParallelLinear / RowParallelLinear under ISP). The LoRA branch is two
small ``nn.Linear`` modules whose weights are *unsharded* (replicated across
WP / TP ranks). Under the shipped 8B config (tp=1, wp=8) this is fine — the
LoRA delta is identical on every rank.

We only target ``fm_modules.*`` (timestep_embedder, fm_head,
vision_model_mot_gen, optional noise_scale_embedder) — see
``modeling_sensenovavl_chat_mot.py`` for where those are constructed.

Inspired by the original LoRA paper (Hu et al., 2022) and PEFT's reference
implementation, simplified to the bits we need here.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LoRALinear: side-branch adapter wrapping an existing nn.Linear.
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """A side-branch LoRA adapter wrapping an existing ``nn.Linear``-like layer.

    The wrapped ``base`` keeps its own forward (including any parallel
    communication). The LoRA branch is two plain ``nn.Linear`` layers, with
    ``lora_B`` initialized to zero so the adapter starts as a no-op.
    """

    # Mark these tensors so external code (savers, optimizers) can detect them.
    _is_lora_module: bool = True

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

        in_features = base.in_features
        out_features = base.out_features
        weight_dtype = base.weight.dtype
        device = base.weight.device

        # Down- and up-projection. Bias-free, the base layer already has bias.
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=weight_dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=weight_dtype)

        # Kaiming-uniform on A, zero on B — standard LoRA init.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Tag LoRA params so create_param_groups / savers can find them by name.
        self.lora_A.weight._is_lora = True  # type: ignore[attr-defined]
        self.lora_B.weight._is_lora = True  # type: ignore[attr-defined]

        # Freeze base — caller is expected to have already done this globally,
        # but we re-assert it locally so accidentally-unfrozen weights here are
        # caught immediately.
        for p in self.base.parameters():
            p.requires_grad = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scale={self.scale:.3f}"

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Some parallel-linear forwards take extra positional/keyword args
        # (sequence_length, cu_seqlens, ...). We forward them transparently
        # to the base, and only run the LoRA branch on the input tensor.
        base_out = self.base(x, *args, **kwargs)

        # If the base returned a tuple (e.g. some MoE / parallel impls do),
        # tack the LoRA delta onto the primary output and pass the rest along.
        if isinstance(base_out, tuple):
            primary, *rest = base_out
            delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
            return (primary + delta, *rest)

        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + delta

    # Convenience accessors that mirror nn.Linear's surface, in case downstream
    # code reaches in and reads them (e.g. shape introspection).
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


# --------------------------------------------------------------------------- #
# Injection helpers.
# --------------------------------------------------------------------------- #
def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    """Match the leaf name against a list of glob-style wildcards."""
    for pat in patterns:
        if pat == "*":
            return True
        # Convert simple shell-like glob to a regex (only '*' is supported).
        regex = "^" + re.escape(pat).replace(r"\*", ".*") + "$"
        if re.match(regex, name):
            return True
    return False


def _set_submodule(root: nn.Module, qualname: str, new_module: nn.Module) -> None:
    """Set ``root.<qualname> = new_module`` given a dotted ``qualname``."""
    parent_path, _, leaf = qualname.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new_module)


def inject_lora(
    model: nn.Module,
    target_prefixes: Sequence[str] = ("fm_modules.",),
    target_leaf_names: Sequence[str] = (
        # Vision-MoT attention (NEOVisionModel)
        "qkv",
        "proj",
        # Standard QKV/O / gate/up/down naming (in case a future config rewires)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Generic MLP layer names found in TimestepEmbedder.mlp and fm_head
        "fc1",
        "fc2",
    ),
    include_sequential_indices: bool = True,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    skip_substrings: Sequence[str] = ("lora_A", "lora_B"),
) -> Tuple[List[str], int, int]:
    """Inject LoRA adapters into every nn.Linear under ``target_prefixes``.

    Args:
        model: the SenseNova-U1 model (after build + pretrained weight loading).
        target_prefixes: only wrap layers whose dotted qualname starts with one
            of these. Defaults to ``fm_modules.`` — i.e. the flow-matching
            branch only.
        target_leaf_names: only wrap layers whose *leaf* attribute name matches
            this set. ``Sequential[idx]`` children appear as ``"0"``/``"2"`` —
            see ``include_sequential_indices``. Use ``("*",)`` to match all.
        include_sequential_indices: also wrap numeric leaf names like ``"0"``,
            ``"2"`` (the Linear layers inside ``TimestepEmbedder.mlp`` and
            ``fm_head`` are addressed this way).
        r: LoRA rank.
        alpha: LoRA scaling numerator (effective scale = alpha / r).
        dropout: LoRA dropout. 0.0 disables it.
        skip_substrings: never wrap a module whose qualname contains one of
            these — used to avoid double-wrapping if the function is called
            twice.

    Returns:
        ``(wrapped_qualnames, num_lora_params, num_frozen_base_params)``.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got r={r}")

    # Step 1: freeze every parameter in the model. Trainable params are added
    # back below for the LoRA branches only.
    n_frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n_frozen += p.numel()

    # Step 2: enumerate target nn.Linear children. Build the list first to
    # avoid mutating the module tree mid-walk.
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        if not any(name.startswith(p) for p in target_prefixes):
            continue

        leaf = name.rsplit(".", 1)[-1]
        leaf_is_numeric = leaf.isdigit()
        matches_name = _matches_any(leaf, target_leaf_names) or (
            include_sequential_indices and leaf_is_numeric
        )
        if not matches_name:
            continue
        candidates.append((name, module))

    if not candidates:
        logger.warning(
            "inject_lora: no nn.Linear matched under prefixes=%s with leaf_names=%s. "
            "LoRA training will have zero trainable params!",
            list(target_prefixes),
            list(target_leaf_names),
        )

    # Step 3: wrap.
    wrapped_names: List[str] = []
    n_lora = 0
    for name, linear in candidates:
        wrapper = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, wrapper)
        wrapped_names.append(name)
        n_lora += wrapper.lora_A.weight.numel() + wrapper.lora_B.weight.numel()

    # Step 4: explicitly mark LoRA params as trainable (they default to True on
    # creation, but be defensive in case the global freeze above ran *after*
    # the wrappers were built in some reorder).
    for p_name, p in model.named_parameters():
        if "lora_A" in p_name or "lora_B" in p_name:
            p.requires_grad = True

    logger.info(
        "inject_lora: wrapped %d Linear layers under %s; "
        "trainable LoRA params=%d, frozen base params=%d (rank=%d, alpha=%d, dropout=%g).",
        len(wrapped_names),
        list(target_prefixes),
        n_lora,
        n_frozen,
        r,
        alpha,
        dropout,
    )
    return wrapped_names, n_lora, n_frozen


# --------------------------------------------------------------------------- #
# State-dict helpers used by the LoRA-only checkpoint saver.
# --------------------------------------------------------------------------- #
def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA parameters (lora_A / lora_B) from the model."""
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if (".lora_A." in name) or (".lora_B." in name) or name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")
    }


def load_lora_state_dict(model: nn.Module, state: dict, strict: bool = True) -> None:
    """Load a LoRA-only state dict produced by :func:`lora_state_dict`.

    ``strict`` mirrors ``nn.Module.load_state_dict`` semantics — but applied
    only to LoRA keys.
    """
    own = {k: v for k, v in model.state_dict().items() if (".lora_A." in k) or (".lora_B." in k)}
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"load_lora_state_dict mismatch: missing={missing[:8]}{'...' if len(missing) > 8 else ''}, "
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )

    # Use ``load_state_dict`` with strict=False so we don't error on the base
    # weights — those should stay frozen at their HF-pretrained values.
    incompatible = model.load_state_dict(state, strict=False)
    # Re-raise anything that's a LoRA key mismatch even with strict=False.
    bad_missing = [k for k in incompatible.missing_keys if (".lora_A." in k) or (".lora_B." in k)]
    if strict and bad_missing:
        raise RuntimeError(f"Missing LoRA keys after load: {bad_missing[:8]}")


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total
