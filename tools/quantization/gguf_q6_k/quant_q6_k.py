"""NumPy Q6_K encoder matching GGML's unweighted reference quantizer.

Ported from ggml-quants.c (make_qx_quants and quantize_row_q6_K_ref),
llama.cpp commit 90e3841d9f35a5bf3c112b9d71502bce77de5b32.
https://github.com/ggml-org/llama.cpp/tree/90e3841d9f35a5bf3c112b9d71502bce77de5b32

MIT License
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _nearest_int(values):
    """Use the same float32 rounding operation as GGML's nearest_int."""
    rounded = values + np.float32(12582912.0)
    return (rounded.view(np.int32) & 0x007FFFFF) - 0x00400000


def _group_scales(values):
    """Fit one scale per 16 values, preserving C's accumulation order."""
    maximum = values[np.abs(values).argmax(axis=0), np.arange(values.shape[1])]
    active = np.abs(maximum) >= np.float32(1e-15)
    maximum = np.where(active, maximum, np.float32(1))
    weights = values * values
    weighted_values = weights * values
    scale = np.zeros(values.shape[1], dtype=np.float32)
    best = np.zeros_like(scale)
    best_inverse = np.zeros_like(scale)
    for step in (0, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        inverse = -(np.float32(32) + np.float32(0.1) * np.float32(step)) / maximum
        levels = np.clip(_nearest_int(values * inverse), -32, 31).astype(np.float32)
        sum_lx = np.zeros_like(scale)
        sum_l2 = np.zeros_like(scale)
        for i in range(16):
            sum_lx += weighted_values[i] * levels[i]
            sum_l2 += (weights[i] * levels[i]) * levels[i]
        candidate = np.divide(sum_lx, sum_l2, out=np.zeros_like(scale), where=sum_l2 != 0)
        if step == 0:
            scale = candidate
            best = scale * sum_lx
            best_inverse = inverse
        else:
            better = (sum_l2 > 0) & (sum_lx * sum_lx > best * sum_l2)
            scale = np.where(better, candidate, scale)
            best = np.where(better, scale * sum_lx, best)
            best_inverse = np.where(better, inverse, best_inverse)
    levels = np.clip(_nearest_int(values * best_inverse), -32, 31) + 32
    return np.where(active, scale, np.float32(0)), np.where(active, levels, 0).astype(np.uint8)


def _quantize_blocks(blocks):
    """Encode blocks of 256 floats into 210-byte Q6_K blocks."""
    count = blocks.shape[0]
    values = np.ascontiguousarray(blocks.reshape(-1, 16).T)
    scales, levels = _group_scales(values)
    scales = scales.reshape(count, 16)
    levels = levels.T.reshape(count, 16, 16)
    maximum = scales[np.arange(count), np.abs(scales).argmax(axis=1)]
    active = np.abs(maximum) >= np.float32(1e-15)
    inverse = np.divide(np.float32(-128), maximum, out=np.zeros_like(maximum), where=active)
    d = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=active).astype("<f2")
    subscales = np.minimum(127, _nearest_int(scales * inverse[:, None])).astype(np.int8)
    actual_scale = d.astype(np.float32)[:, None] * subscales.astype(np.float32)
    ratios = np.divide(
        blocks.reshape(count, 16, 16),
        actual_scale[:, :, None],
        out=np.zeros((count, 16, 16), dtype=np.float32),
        where=actual_scale[:, :, None] != 0,
    )
    rounded = (np.clip(_nearest_int(ratios), -32, 31) + 32).astype(np.uint8)
    levels = np.where(actual_scale[:, :, None] != 0, rounded, levels).reshape(count, 2, 4, 32)
    q1, q2, q3, q4 = (levels[:, :, i] for i in range(4))
    ql = np.stack(((q1 & 15) | ((q3 & 15) << 4), (q2 & 15) | ((q4 & 15) << 4)), axis=2)
    qh = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    packed = np.concatenate(
        (ql.reshape(count, 128), qh.reshape(count, 64), subscales.view(np.uint8), d.reshape(count, 1).view(np.uint8)),
        axis=1,
    )
    packed[~active] = 0
    return packed


def quantize_q6(array, qtype):
    """Quantize a float32 matrix without a native library or compiler."""
    if int(qtype) != 14:
        raise ValueError("This encoder only supports Q6_K")
    array = np.asarray(array, dtype=np.float32, order="C")
    if array.ndim != 2 or array.shape[1] % 256:
        raise ValueError("Q6_K requires a matrix whose row width is divisible by 256")
    rows, cols = array.shape
    blocks = array.reshape(-1, 256)
    output = np.empty((len(blocks), 210), dtype=np.uint8)
    chunk_size = 2048
    workers = min(8, max(1, (len(blocks) + chunk_size - 1) // chunk_size))

    def encode(worker):
        for start in range(worker * chunk_size, len(blocks), workers * chunk_size):
            chunk = blocks[start : start + chunk_size]
            if not np.isfinite(chunk).all():
                raise ValueError("Q6_K input contains nonfinite values")
            output[start : start + len(chunk)] = _quantize_blocks(chunk)

    if workers == 1:
        encode(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(encode, range(workers)))
    return output.reshape(rows, cols // 256 * 210)
