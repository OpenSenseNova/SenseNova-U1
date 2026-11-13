"""Call GGML's Q6_K encoder; the shared library stays beside this file."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gguf
import numpy as np

LIB_PATH = Path(__file__).with_name("libggml-base.so")
if not LIB_PATH.is_file():
    raise FileNotFoundError(f"Build the Q6 encoder first: bash {LIB_PATH.with_name('build_ggml.sh')}")
LIB = ctypes.CDLL(str(LIB_PATH))
LIB.ggml_quantize_chunk.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
]
LIB.ggml_quantize_chunk.restype = ctypes.c_size_t


def quantize_q6(array, qtype):
    array = np.ascontiguousarray(array, dtype=np.float32)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    rows, cols = array.shape
    out = np.empty((rows, cols // block * size), dtype=np.uint8)

    def encode(start):
        end = min(rows, start + stride)
        count = LIB.ggml_quantize_chunk(
            int(qtype), array[start:end].ctypes.data, out[start:end].ctypes.data, 0, end - start, cols, None
        )
        if count != out[start:end].nbytes:
            raise RuntimeError("GGML output size mismatch")

    stride = max(1, (rows + 7) // 8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(encode, range(0, rows, stride)))
    return out
