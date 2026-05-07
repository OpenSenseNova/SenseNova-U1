"""Layer offload wrapper for memory-efficient inference.

Keeps each layer of an ``nn.ModuleList`` in CPU pinned memory and moves it
onto a CUDA device on demand. Two modes share a single :class:`LayerOffloadWrapper`:

- ``prefetch_count == 0`` — synchronous: load before forward, evict after.
- ``prefetch_count >= 1`` — asynchronous: a dedicated CUDA stream prefetches
  the next ``prefetch_count`` layers so the H2D copy overlaps compute.

General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (``transformer_blocks``, ``layers``, …). Each
layer is evicted back to CPU immediately after its forward completes; in
async mode prefetch wraps around modulo the layer count so the last layer's
prefetch warms up early layers for the next forward pass.

Inference-only — the eviction-after-forward design destroys gradient flow,
so :meth:`__init__` rejects models in training mode.

Origin: adapted from `Lightricks/LTX-2 <https://github.com/Lightricks/LTX-2>`_.

Example
-------
>>> model = build_my_model(device=torch.device("cpu")).eval()
>>> model = LayerOffloadWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)
>>> model.teardown()
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _log_vram(label: str, target_device: torch.device, *, reset_peak: bool = False) -> None:
    """Cheap VRAM snapshot for diagnosing offload-mode leaks across repeated
    runs (notably under ComfyUI). Never raises; logs at INFO so it shows up
    without explicit DEBUG opt-in.
    """
    try:
        if target_device.type != "cuda" or not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated(target_device) / (1024**3)
        reserved = torch.cuda.memory_reserved(target_device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(target_device) / (1024**3)
        logger.info(
            "[layer_offload vram] %-40s | alloc=%6.2f GiB  reserved=%6.2f GiB  peak=%6.2f GiB",
            label,
            alloc,
            reserved,
            peak,
        )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(target_device)
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.debug("vram log %r failed: %s", label, exc)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def _require_cuda(target_device: torch.device) -> None:
    """Reject non-CUDA target devices with a clear error.

    Pinned memory, dedicated streams, and ``record_stream`` are CUDA-only
    APIs in PyTorch; ROCm reuses the ``cuda`` namespace, so it works, but
    MPS / XPU / CPU do not. Failing fast here is more honest than letting
    the wrapper appear to accept any device when ``target_device`` is in
    fact a façade.
    """
    if target_device.type != "cuda":
        raise NotImplementedError(
            f"Layer offload requires a CUDA target_device (got {target_device!r}). "
            "Pinned memory and prefetch streams are CUDA-specific; MPS / XPU / CPU "
            "are not supported."
        )


def _is_cuda_malloc_async_backend() -> bool:
    """Detect whether the active CUDA caching allocator is ``cudaMallocAsync``.

    The native caching allocator and ``cudaMallocAsync`` differ on a point
    that matters for our cross-stream prefetch: ``cudaMallocAsync`` keeps a
    pool *per stream* and never reuses freed blocks across streams without
    explicit ordering, so allocating on the prefetch stream and freeing on
    the compute stream causes the reserved pool to grow without bound. The
    native allocator handles this case with ``record_stream`` and reuses
    blocks freely.

    ComfyUI launchers commonly set
    ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync``; standalone Python
    runs typically don't.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.get_allocator_backend() == "cudaMallocAsync"
    except Exception:
        return False


def _audit_lazy_state(
    model: nn.Module,
    target_device: torch.device,
    managed_tensor_ids: set[int],
) -> int:
    """Move any params/buffers stranded off ``target_device`` after the first
    forward (lazy buffers materialised inside ``forward()``) onto it.

    Returns the number of tensors moved. Tensors already managed by the
    offload store are skipped — they are intentionally rotated between
    pinned CPU and GPU. Anything else that ends up on the wrong device is
    almost certainly a lazy buffer (e.g. an attention mask cache) that the
    constructor could not see, and it lives on GPU permanently from here on.
    """
    moved = 0
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if id(tensor) in managed_tensor_ids:
            continue
        if tensor.device != target_device:
            tensor.data = tensor.data.to(target_device)
            moved += 1
    return moved


class _LayerStore:
    """Holds CPU-pinned copies of every parameter/buffer of every offloaded layer.

    Tracks which layers currently reside on GPU so the prefetcher and evictor
    can make correct decisions in async mode. In sync mode the bookkeeping is
    free overhead.
    """

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        self._pinned: list[dict[str, torch.Tensor]] = []
        self._on_gpu: set[int] = set()

        for layer in layers:
            pinned: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                pinned_tensor = tensor.data.pin_memory()
                tensor.data = pinned_tensor
                pinned[name] = pinned_tensor
            self._pinned.append(pinned)

    def _check_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def move_to_gpu(self, idx: int, layer: nn.Module, *, non_blocking: bool = False) -> None:
        """Move layer *idx* parameters from pinned CPU to ``target_device``."""
        self._check_idx(idx)
        if idx in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in pinned:
                param.data = pinned[name].to(self.target_device, non_blocking=non_blocking)
        self._on_gpu.add(idx)

    def evict_to_cpu(self, idx: int, layer: nn.Module) -> None:
        """Swap layer *idx* parameters back to their pinned CPU copies."""
        self._check_idx(idx)
        if idx not in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in pinned:
                param.data = pinned[name]
        self._on_gpu.discard(idx)

    def managed_tensor_ids(self) -> set[int]:
        ids: set[int] = set()
        for pinned in self._pinned:
            for t in pinned.values():
                ids.add(id(t))
        return ids

    def cleanup(self) -> None:
        """Drop the pinned-tensor refs so they can be freed by the GC."""
        for pinned_dict in self._pinned:
            pinned_dict.clear()
        self._pinned.clear()
        self._on_gpu.clear()


class _AsyncPrefetcher:
    """Issues H2D transfers on a dedicated CUDA stream.

    Uses per-layer CUDA events so that the compute stream only waits for the
    specific layer it needs, not all pending transfers.
    """

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        """Begin async transfer of layer *idx* to GPU (no-op if already there)."""
        if self._store.is_on_gpu(idx) or idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event

    def wait(self, idx: int) -> None:
        """Block the compute stream until layer *idx*'s transfer completes."""
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        """Drain pending work and release CUDA stream/event resources."""
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None


class LayerOffloadWrapper(nn.Module):
    """Wraps a model to offload its sequential layers between CPU and GPU.

    Each layer is evicted immediately after its forward completes. With
    ``prefetch_count == 0`` the wrapper runs in synchronous mode (one layer
    on GPU at a time, no extra stream). With ``prefetch_count >= 1`` it
    pre-stages the next layers on a dedicated CUDA stream so H2D overlaps
    compute, with up to ``1 + prefetch_count`` layers resident on GPU.

    Parameters
    ----------
    model:
        The model to wrap, with all parameters on **CPU** and in eval mode.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential layers
        (e.g. ``"transformer_blocks"`` or ``"language_model.model.layers"``).
    target_device:
        The CUDA device to use for compute. MPS / XPU / CPU are rejected.
    prefetch_count:
        ``0`` = synchronous (per-layer load/evict, lowest VRAM, slowest).
        ``>= 1`` = async prefetch this many layers ahead (faster, more VRAM).
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 0,
    ) -> None:
        super().__init__()
        _require_cuda(target_device)
        if prefetch_count < 0:
            raise ValueError("prefetch_count must be >= 0")
        if model.training:
            raise RuntimeError(
                "LayerOffloadWrapper only supports inference; the per-forward "
                "evict-to-CPU step destroys gradient flow. Call model.eval() first."
            )

        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        # Clamp: no point prefetching more layers than (num_layers - 1).
        max_prefetch = max(len(self._layers) - 1, 0)
        self._prefetch_count = min(prefetch_count, max_prefetch)
        self._async_mode = self._prefetch_count >= 1
        # ``cudaMallocAsync`` keeps per-stream memory pools and never reuses
        # freed blocks across streams without explicit ordering. Detect the
        # backend at construction time so the hooks can pick the right
        # alloc/free pairing strategy: native allocator → record_stream
        # (fast, frees go to whatever stream is current); cudaMallocAsync →
        # wait_stream + free on prefetch stream (correct, slightly more
        # serialized).
        self._cuda_malloc_async = _is_cuda_malloc_async_backend()
        if self._async_mode:
            logger.info(
                "LayerOffloadWrapper: async prefetch enabled (prefetch_count=%d, allocator=%s, free_path=%s)",
                self._prefetch_count,
                "cudaMallocAsync" if self._cuda_malloc_async else "native",
                "prefetch-stream + wait_stream" if self._cuda_malloc_async else "compute-stream + record_stream",
            )
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._audit_handle: torch.utils.hooks.RemovableHandle | None = None
        self._prefetcher: _AsyncPrefetcher | None = None

        _log_vram("wrapper.__init__: pre-setup", target_device, reset_peak=True)
        self._setup()
        _log_vram(
            f"wrapper.__init__: post-setup (async={self._async_mode}, "
            f"prefetch={self._prefetch_count}, layers={len(self._layers)})",
            target_device,
        )

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        # 1. Pin all layer tensors in CPU memory.
        self._store = _LayerStore(self._layers, self._target_device)

        # 2. Move all NON-layer params/buffers to GPU permanently.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        # 3. In async mode: pre-load the first (1 + prefetch_count) layers and
        #    spin up the prefetch stream.
        if self._async_mode:
            for idx in range(min(self._prefetch_count + 1, len(self._layers))):
                self._store.move_to_gpu(idx, self._layers[idx])
            self._prefetcher = _AsyncPrefetcher(self._store, self._layers)

        # 4. Register layer load/evict hooks.
        self._register_hooks()

        # 5. One-shot audit: catch lazy params/buffers materialised inside the
        #    first forward (RoPE caches, attention masks, etc.) that escaped
        #    the construction-time scan.
        self._audit_handle = self._model.register_forward_hook(self._audit_first_forward)

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        def _pre_hook(module: nn.Module, _args: Any, *, idx: int) -> None:
            if self._async_mode:
                # Wait only for THIS layer's H2D transfer.
                self._prefetcher.wait(idx)  # type: ignore[union-attr]
                if not self._store.is_on_gpu(idx):
                    self._store.move_to_gpu(idx, module)

                if not self._cuda_malloc_async:
                    # Native caching allocator fast path: tell the allocator
                    # the compute stream will read these weights so it does
                    # not reuse the blocks while the kernel is still running.
                    # Frees in _post_hook go to whatever stream is current
                    # (compute stream) and the allocator handles cross-stream
                    # reuse internally — no prefetch-stream barrier needed.
                    compute_stream = torch.cuda.current_stream(self._target_device)
                    for param in itertools.chain(module.parameters(), module.buffers()):
                        param.data.record_stream(compute_stream)

                # Kick off prefetch for upcoming layers (wraps around for next pass).
                for offset in range(1, self._prefetch_count + 1):
                    self._prefetcher.prefetch((idx + offset) % num_layers)  # type: ignore[union-attr]
            else:
                # Sync mode: the H2D dispatches on the compute stream itself,
                # which serialises naturally with the kernel that follows.
                self._store.move_to_gpu(idx, module, non_blocking=True)

        def _post_hook(module: nn.Module, _args: Any, _output: Any, *, idx: int) -> None:
            if self._async_mode and self._cuda_malloc_async:
                # cudaMallocAsync slow-but-safe path: per-stream pools
                # require alloc and free on the same stream. Since
                # `_AsyncPrefetcher` allocates layer weights on the prefetch
                # stream, we must also free them there. Wait for the compute
                # stream to finish reading the weights first; wait_stream is
                # host-async so it does not stall Python. The cost is that
                # subsequent prefetches queued on the prefetch stream are
                # ordered after this wait, slightly reducing pipeline depth.
                prefetch_stream = self._prefetcher._stream  # type: ignore[union-attr]
                compute_stream = torch.cuda.current_stream(self._target_device)
                prefetch_stream.wait_stream(compute_stream)
                with torch.cuda.stream(prefetch_stream):
                    self._store.evict_to_cpu(idx, module)
            else:
                # Native allocator path: just drop the GPU tensor refs on
                # the compute stream. record_stream in _pre_hook ensures the
                # blocks are not reused before the kernel finishes.
                self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def _audit_first_forward(self, _module: nn.Module, _inputs: Any, _outputs: Any) -> None:
        _log_vram("wrapper.audit: pre", self._target_device)
        moved = _audit_lazy_state(self._model, self._target_device, self._store.managed_tensor_ids())
        if moved:
            logger.warning(
                "LayerOffloadWrapper: moved %d lazy param(s)/buffer(s) onto %s after "
                "the first forward. These will stay on GPU; they are not offloaded.",
                moved,
                self._target_device,
            )
        _log_vram(f"wrapper.audit: post (moved={moved})", self._target_device)
        if self._audit_handle is not None:
            self._audit_handle.remove()
            self._audit_handle = None

    def teardown(self) -> None:
        """Remove hooks, release pinned memory, and move parameters back to CPU.

        After this call the wrapper is inert: hooks are removed, the prefetch
        stream is drained and destroyed, all parameters reside on regular
        (non-pinned) CPU memory, and the :class:`_LayerStore` pinned-tensor
        cache is cleared.
        """
        _log_vram(
            f"wrapper.teardown: enter (on_gpu={len(self._store._on_gpu)}, "
            f"events={len(self._prefetcher._events) if self._prefetcher is not None else 0})",
            self._target_device,
        )
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._audit_handle is not None:
            self._audit_handle.remove()
            self._audit_handle = None

        # Drain in-flight H2D copies before tearing down stream resources, or
        # the CUDA driver can hit use-after-free during cleanup.
        torch.cuda.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        self._store.cleanup()
        _log_vram("wrapper.teardown: exit (pre-empty_cache)", self._target_device)

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped model.

        ``nn.Module.__getattr__`` is only called when normal lookup fails, so
        ``_model`` / ``_store`` etc. are still resolved via ``__dict__``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
