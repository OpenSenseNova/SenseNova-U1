"""Inference timing profiler.

Records model-load time and per-generation wall time (CUDA-synchronized so
GPU launch overhead doesn't hide inside Python). ``report()`` prints a summary
that also converts per-image time into per-token cost using a fixed image
patch size (the model's generation patchification factor). For CUDA devices,
it also records peak memory allocated/reserved during model load and each
generation block.

Intended for quick, human-readable profiling from CLI scripts under
``examples/``. When ``enabled=False``, every context manager is a no-op and
``report()`` prints nothing, so it can be wired in unconditionally.

Typical usage::

    from sensenova_u1.utils import InferenceProfiler

    prof = InferenceProfiler(enabled=args.profile, device=args.device)
    with prof.time_load():
        engine = SenseNovaU1T2I(model_path)
    with prof.time_generate(width=2048, height=2048, batch=1):
        images = engine.generate(...)
    prof.report()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List

import torch

DEFAULT_IMAGE_PATCH_SIZE = 32


@dataclass
class _MemoryPeak:
    allocated: int = 0
    reserved: int = 0
    by_device: tuple[tuple[str, int, int], ...] = ()

    @property
    def available(self) -> bool:
        return self.allocated > 0 or self.reserved > 0


@dataclass
class _GenerationRecord:
    width: int
    height: int
    batch: int
    seconds: float
    memory_peak: _MemoryPeak


class InferenceProfiler:
    """Minimal wall-clock profiler for model loading + generation.

    Parameters
    ----------
    enabled : bool
        If False, every method is a no-op (zero overhead).
    device : str
        E.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``. Used to decide whether to
        ``torch.cuda.synchronize()`` around timed blocks.
    patch_size : int, optional
        Image-token grid factor used by :meth:`report` to translate wall time
        into ms/token. Defaults to :data:`DEFAULT_IMAGE_PATCH_SIZE`.
    """

    def __init__(
        self,
        enabled: bool,
        device: str = "cuda",
        patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.patch_size = patch_size
        self.load_time: float = 0.0
        self.load_memory_peak = _MemoryPeak()
        self.gen_records: List[_GenerationRecord] = []

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.enabled and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _has_cuda_memory_stats(self) -> bool:
        return self.enabled and self.device.startswith("cuda") and torch.cuda.is_available()

    def _cuda_devices(self) -> list[torch.device]:
        device = torch.device(self.device)
        if device.type != "cuda":
            return []
        if device.index is not None:
            return [device]
        return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]

    def _reset_memory_peak(self) -> None:
        if self._has_cuda_memory_stats():
            for device in self._cuda_devices():
                torch.cuda.reset_peak_memory_stats(device)

    def _memory_peak(self) -> _MemoryPeak:
        if not self._has_cuda_memory_stats():
            return _MemoryPeak()
        by_device = tuple(
            (
                str(device),
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            )
            for device in self._cuda_devices()
        )
        return _MemoryPeak(
            allocated=sum(allocated for _, allocated, _ in by_device),
            reserved=sum(reserved for _, _, reserved in by_device),
            by_device=by_device,
        )

    @contextmanager
    def time_load(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        self._reset_memory_peak()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.load_time = time.perf_counter() - t0
            self.load_memory_peak = self._memory_peak()

    @contextmanager
    def time_generate(self, width: int, height: int, batch: int = 1) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        self._reset_memory_peak()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.gen_records.append(
                _GenerationRecord(
                    width=width,
                    height=height,
                    batch=batch,
                    seconds=time.perf_counter() - t0,
                    memory_peak=self._memory_peak(),
                )
            )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a summary. No-op when ``enabled=False``."""
        if not self.enabled:
            return
        print()
        print("=" * 64)
        print("Profile summary")
        print("=" * 64)
        print(f"  model load          : {self.load_time:8.3f} s")
        if self.load_memory_peak.available:
            print(f"  load peak memory    : {self._format_memory(self.load_memory_peak)}")
        if not self.gen_records:
            print("  (no generations were timed)")
            return

        total_images = sum(record.batch for record in self.gen_records)
        total_time = sum(record.seconds for record in self.gen_records)
        avg_per_image = total_time / total_images

        total_tokens = sum(
            (record.width // self.patch_size) * (record.height // self.patch_size) * record.batch
            for record in self.gen_records
        )
        avg_tokens = total_tokens / total_images
        tokens_per_sec = total_tokens / total_time

        peak_generation_memory = self._max_memory_peak(record.memory_peak for record in self.gen_records)

        print(
            f"  generations         : {len(self.gen_records)} call(s), "
            f"{total_images} image(s) total, {total_time:.3f} s wall"
        )
        print(f"  avg per image       : {avg_per_image:8.3f} s")
        print(
            f"  image tokens        : patch_size={self.patch_size}, "
            f"avg {avg_tokens:.0f} tok/image ({int(avg_tokens):d})"
        )
        print(f"  throughput          : {tokens_per_sec:8.2f} tok/s")
        if peak_generation_memory.available:
            print(f"  generation peak mem : {self._format_memory(peak_generation_memory)}")

        if len(self.gen_records) > 1:
            print("  per-call breakdown  :")
            for idx, record in enumerate(self.gen_records):
                tokens = (record.width // self.patch_size) * (record.height // self.patch_size) * record.batch
                memory = f", {self._format_memory(record.memory_peak)}" if record.memory_peak.available else ""
                print(
                    f"    [{idx + 1:>3}] {record.width}x{record.height} x{record.batch}  "
                    f"{record.seconds:7.3f} s  ({tokens:>6d} tok, "
                    f"{tokens / record.seconds:8.2f} tok/s{memory})"
                )
        print("=" * 64)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        return f"{num_bytes / (1024**3):.2f} GiB"

    @classmethod
    def _format_memory(cls, memory_peak: _MemoryPeak) -> str:
        text = (
            f"allocated {cls._format_bytes(memory_peak.allocated)}, reserved {cls._format_bytes(memory_peak.reserved)}"
        )
        if len(memory_peak.by_device) <= 1:
            return text
        details = ", ".join(
            f"{name}: {cls._format_bytes(allocated)} alloc/{cls._format_bytes(reserved)} reserved"
            for name, allocated, reserved in memory_peak.by_device
        )
        return f"{text} ({details})"

    @staticmethod
    def _max_memory_peak(memory_peaks: Iterator[_MemoryPeak]) -> _MemoryPeak:
        max_peak = _MemoryPeak()
        by_device: dict[str, tuple[int, int]] = {}
        for memory_peak in memory_peaks:
            max_peak.allocated = max(max_peak.allocated, memory_peak.allocated)
            max_peak.reserved = max(max_peak.reserved, memory_peak.reserved)
            for name, allocated, reserved in memory_peak.by_device:
                prev_allocated, prev_reserved = by_device.get(name, (0, 0))
                by_device[name] = (max(prev_allocated, allocated), max(prev_reserved, reserved))
        max_peak.by_device = tuple((name, allocated, reserved) for name, (allocated, reserved) in by_device.items())
        return max_peak
