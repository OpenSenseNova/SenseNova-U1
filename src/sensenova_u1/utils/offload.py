"""Context managers for CPU<->GPU layer offload during inference.

Wraps :class:`LayerOffloadWrapper` from :mod:`.layer_offload` so callers can
enter a ``with`` block, run generation through the wrapped model, and have
the wrapper torn down + host pinned-memory cache released on exit.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TypeVar

import torch
from torch import nn

from .layer_offload import LayerOffloadWrapper

LOGGER = logging.getLogger(__name__)

_M = TypeVar("_M", bound=nn.Module)

VRAM_MODE_OPTIONS: tuple[str, ...] = ("full", "low", "balanced")
DEFAULT_VRAM_MODE: str = "full"
_VRAM_MODE_TO_PREFETCH: dict[str, int] = {
    "full": 0,
    "low": 1,
    "balanced": 2,
}
DEFAULT_LAYERS_ATTR: str = "language_model.model.layers"


def vram_mode_to_prefetch_count(mode: str) -> int:
    """Map a ``--vram_mode`` choice to the layer-offload ``prefetch_count``.

    ``0`` means the model stays fully on GPU (no offload). ``1`` means
    synchronous per-layer swap; ``>=2`` means async prefetch.
    """
    if mode not in _VRAM_MODE_TO_PREFETCH:
        raise ValueError(f"Unsupported vram_mode={mode!r}. Choose one of {VRAM_MODE_OPTIONS}.")
    return _VRAM_MODE_TO_PREFETCH[mode]


def make_offload_ctx(
    model: nn.Module,
    prefetch_count: int,
    target_device: str | torch.device,
    layers_attr: str = DEFAULT_LAYERS_ATTR,
) -> AbstractContextManager[nn.Module]:
    """Pick the right offload context for ``prefetch_count``.

    ``0`` returns a pass-through context yielding ``model`` unchanged.
    ``1`` returns the synchronous offload context (one resident layer).
    ``>=2`` returns the async prefetch context with that many layers ahead.
    """
    if prefetch_count == 0:
        return contextlib.nullcontext(model)
    target = target_device if isinstance(target_device, torch.device) else torch.device(target_device)
    if prefetch_count == 1:
        return offload_layers_sync(model, layers_attr, target)
    return offload_layers_async(model, layers_attr, target, prefetch_count=prefetch_count)


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _log_vram(label: str, target_device: torch.device) -> None:
    """Log allocated / reserved / peak VRAM with ``label``.

    Used to diagnose the ComfyUI-only VRAM growth under
    ``vram_mode='balanced'``. Best-effort; never raises.
    """
    try:
        if target_device.type != "cuda" or not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated(target_device) / (1024**3)
        reserved = torch.cuda.memory_reserved(target_device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(target_device) / (1024**3)
        LOGGER.info(
            "[offload vram] %-40s | alloc=%6.2f GiB  reserved=%6.2f GiB  peak=%6.2f GiB",
            label,
            alloc,
            reserved,
            peak,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.debug("offload vram log %r failed: %s", label, exc)


def _empty_host_cache(target_device: torch.device) -> None:
    """Release PyTorch's pinned host-memory cache.

    Without this, repeated offload runs eventually exhaust host memory
    because the CachingHostAllocator keeps freed pinned blocks cached.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device=target_device)
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        LOGGER.warning("offload: host cache release failed: %s", exc)


@contextmanager
def _offload_layers(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[nn.Module]:
    wrapper = LayerOffloadWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        prefetch_count=prefetch_count,
    )
    try:
        yield wrapper
    finally:
        try:
            wrapper.teardown()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("offload: teardown failed: %s", exc)
        try:
            model.to("cpu")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("offload: model.to('cpu') failed: %s", exc)
        _log_vram("offload._offload_layers: pre-empty_cache", target_device)
        _cleanup_memory()
        _log_vram("offload._offload_layers: post-empty_cache", target_device)
        _empty_host_cache(target_device)
        _log_vram("offload._offload_layers: post-host_empty_cache", target_device)


def offload_layers_sync(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
) -> AbstractContextManager[nn.Module]:
    """Synchronous CPU<->GPU layer offload. Lower memory, slower.

    Each offloaded layer is loaded just before its forward and evicted right
    after; exactly one layer's weights are resident on GPU.
    """
    return _offload_layers(model, layers_attr, target_device, prefetch_count=0)


def offload_layers_async(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int = 2,
) -> AbstractContextManager[nn.Module]:
    """Async-prefetch layer offload. Higher memory, faster.

    ``prefetch_count`` is how many layers ahead to prefetch on a dedicated
    CUDA stream; must be >= 1.
    """
    if prefetch_count < 1:
        raise ValueError("prefetch_count must be >= 1 for async offload")
    return _offload_layers(model, layers_attr, target_device, prefetch_count=prefetch_count)
