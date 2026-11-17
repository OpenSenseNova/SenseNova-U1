"""INT8 + ConvRot exporter for ComfyUI — auto layer detection.

Quantizes the per-token block linears (attention + FFN), passes everything else through.
Recipe: fp32 upcast, block-Hadamard rotation at the per-layer best power-of-4 groupsize,
per-channel absmax scale, embedded `<layer>.comfy_quant` config.

    python convert_int8_convrot.py SRC [DST.safetensors] [--dry-run] [options]

SRC: .safetensors (lazy) or torch pickle .pth/.pt/.ckpt (safe weights_only load, held in RAM).
DST optional: defaults to SRC with bf16/fp16/fp32 -> int8_convrot (or _int8_convrot appended).
Auto-detect can't see token count M (small-M/windowed/audio layers get quantized anyway — size
win, maybe not speed) or loader quirks (manual_cast/key-remap loaders -> loads but outputs garbage).
Run --dry-run on an unfamiliar architecture and use --min-gemm / --exclude as
needed. The output uses ComfyUI's per-module ``.comfy_quant`` format and is
not the file-level format emitted by the native SenseNova converter.
"""

# ruff: noqa: T201  (print() is this CLI's output)
import argparse
import collections
import json
import os
import re
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
except ImportError:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

VALID_GS = (256, 64, 16)  # convrot Hadamard sizes; power-of-4, prefer largest
CLIP_GRID = torch.linspace(0.55, 1.0, 80)
FP8 = (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))


def best_gs(k):
    return next((g for g in VALID_GS if k % g == 0), None)


# Token-embedding lookup tables -> per-row int8 (+rotation). Match real token tables
# (embed_tokens / embed_tokens_per_layer / wte / ...), not an `*.embedding_projection`.
EMBED_SEG = re.compile(r"^(embed_tokens(_per_layer)?|token_embedding|word_embeddings|tok_embeddings|wte|shared)$")


def is_token_embedding(key, shape):
    if len(shape) != 2:
        return False
    n, k = shape
    if n < 4096 or n <= k:  # vocab-sized table (rows >> cols)
        return False
    return any(EMBED_SEG.match(s) for s in key.split("."))


# safe_open-compatible reader for torch pickle checkpoints (weights_only=True -> safe load, no code
# execution; whole file into RAM since pickle has no lazy access).
_DTYPE_CODE = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    getattr(torch, "float8_e5m2", None): "F8_E5M2",
}


class _TorchSlice:  # mimics safetensors get_slice()
    def __init__(self, t):
        self._t = t

    def get_shape(self):
        return list(self._t.shape)

    def get_dtype(self):
        return _DTYPE_CODE.get(self._t.dtype, str(self._t.dtype))


class _TorchReader:
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd = self._find_state_dict(obj)
        if sd is None:
            raise ValueError(f"no tensor state-dict found in {path}")
        self._sd = sd

    @staticmethod
    def _find_state_dict(obj):
        if isinstance(obj, dict):
            tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors  # this level holds the weights
            for key in ("state_dict", "model_state_dict", "model", "module", "net", "ema", "params"):
                if isinstance(obj.get(key), dict):
                    found = _TorchReader._find_state_dict(obj[key])
                    if found:
                        return found
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metadata(self):
        return {}

    def keys(self):
        return list(self._sd.keys())

    def get_slice(self, k):
        return _TorchSlice(self._sd[k])

    def get_tensor(self, k):
        return self._sd[k]


def open_model(path):
    """safe_open for .safetensors; safe (weights_only) torch.load for .pth/.pt/.ckpt/.bin."""
    if path.lower().endswith(".safetensors"):
        return safe_open(path, framework="pt", device="cpu")
    return _TorchReader(path)


# Detection = quantize every eligible 2-D block linear, minus a name denylist. No projection-name
# allowlist (fragile: every arch invents new names like to_qkv/add_q_proj/single_blocks.linear1).
# adaLN MODULATION *is* quantized (it's a big M=batch GEMM that quantizes cleanly ~0.9%, and on
# Qwen/Flux it's 18-33% of the model — no reason to leave it bf16). We only exclude what's genuinely
# not a per-token GEMM worth quantizing: scale_shift buffers, rope/pos_embed, input embedders,
# gate/router logits, M=1 timestep MLPs, output head/final. 1-D norms are dropped by not-2d already.
# Careful bits: `embedder` (not `embed`) keeps `*_embeddings_connector` in; bare `gate` stays
# (SwiGLU) — only gate_logits/router drop; `timestep`/`time` catch the M=1 embed but not the modulator.
EXCLUDE_SEG = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)"
)
# `refiner_blocks` = short-M text side-path (Krea txtfusion.refiner_blocks); main-stream refiners are
# `*_refiner` (Boogu/Z-Image), kept. `adapter` = conditioning injection modules (Wan-Animate
# face_adapter, ip/control adapters): tiny, identity/quality-critical, quantize worst -> leave bf16.


def classify(key, shape):
    """Quantize every eligible 2-D block linear except the name denylist. Returns (bool, reason)."""
    if len(shape) != 2:
        return (False, "not-2d")
    n, k = shape
    if n < 8:
        return (False, "small-N")
    gs = best_gs(k)
    if gs is None:
        return (False, "ineligible-K")
    segs = key.split(".")
    # in a block = an integer segment with named structure after it (blocks.5.attn.q). A trailing
    # integer is a Sequential index on a top-level MLP (tmlp.0, img_emb.proj.1) -> not a block.
    if not any(segs[i].isdigit() for i in range(len(segs) - 1)):
        return (False, "not-in-indexed-block")
    if any(EXCLUDE_SEG.search(s) for s in segs):
        return (False, "denylist(scale_shift/embed/gate/time/head/refiner_blocks/adapter)")
    return (True, f"gs{gs}")


# ---------------------------------------------------------------------------
# Quantization (fp32 upcast + block-Hadamard rotation + MSE-optimal per-channel clip)
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_convrot(w, gs, mseclip=True, device="cuda"):
    wf = w.to(device, torch.float32)
    h = _build_hadamard(gs, device=wf.device, dtype=torch.float32)
    wr = _rotate_weight(wf, h, gs)
    absmax = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mseclip:
        scale = (absmax / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        return q.to(torch.int8), scale.to(torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = None
    for a in CLIP_GRID.tolist():
        scale = (absmax * a / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        mse = ((q * scale - wr) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q.clone() if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q.to(torch.int8), best_scale.to(torch.float32)


@torch.no_grad()
def recon_metrics(qd, scale, w_ref, gs, device="cuda"):
    """Reconstruct (dequant + un-rotate) and return (cosine, relative_error_%)."""
    deq = qd.to(device).float() * scale.to(device)
    h = _build_hadamard(gs, device=device, dtype=torch.float32)
    deq = _rotate_weight(deq, h, gs)
    wf = w_ref.to(device).float()
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), wf.flatten(), dim=0).item()
    relerr = ((deq - wf).norm() / wf.norm().clamp(min=1e-30)).item() * 100.0
    return cos, relerr


def cq_tensor(gs):
    cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)


@torch.no_grad()
def quantize_w4a8(w, bits=4, group_size=None, device="cuda"):
    """W4A8: ConvRot-rotated int4 weight with a Lloyd-Max codebook + fp8 group
    scales, or with bits=6 W6A8: uniform int6, no codebook, ~3x lower weight error for
    1.5x the bytes. Activations quantize to int8 at runtime. Needs K divisible by 256 and N>=64.
    Returns (out-tensor dict, comfy_quant cfg, relerr%)."""
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor

    group_size = group_size or (32 if bits == 6 else 16)
    wf = w.to(device, torch.bfloat16)
    q, p = AsymW4A8Int8Layout.quantize(
        wf,
        group_size=group_size,
        convrot_groupsize=256,
        scale_dtype=torch.float8_e4m3fn,
        codebook=True,
        stochastic_rounding=0,
        bits=bits,
    )
    deq = QuantizedTensor(q, "AsymW4A8Int8Layout", p).dequantize()
    tensors = {"weight": q.cpu(), "weight_s_rel": p.scale.cpu(), "weight_s_channel": p.s_channel.cpu()}
    if p.codebook is not None:
        tensors["weight_codebook"] = p.codebook.cpu()
    cfg = {
        "format": "w6a8_int8" if bits == 6 else "asym_w4a8_int8",
        "group_size": group_size,
        "convrot": True,
        "convrot_groupsize": 256,
    }
    relerr = ((deq.float() - wf.float()).norm() / wf.float().norm().clamp(min=1e-30)).item() * 100.0
    return tensors, cfg, relerr


@torch.no_grad()
def quantize_embedding(w, gs, device="cuda", chunk=32768):
    """Rotated per-row int8 for token-embedding tables. Rotation Gaussianizes each row, tightening
    its absmax (~half the error for free). The runtime un-rotates after the lookup, since a lookup
    has no GEMM to fold the inverse into. gs=None -> plain absmax per-row int8 (no rotation).
    Chunked over rows so a ~1B-element vocab table never lands fully on the GPU."""
    h = _build_hadamard(gs, device=device, dtype=torch.float32) if gs else None
    qs, ss = [], []
    for i in range(0, w.shape[0], chunk):
        wf = w[i : i + chunk].to(device, torch.float32)
        wr = _rotate_weight(wf, h, gs) if h is not None else wf
        scale = (wr.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        qs.append(q.to(torch.int8).cpu())
        ss.append(scale.to(torch.float32).cpu())
        del wf, wr, q, scale
        torch.cuda.empty_cache()
    return torch.cat(qs), torch.cat(ss)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument(
        "dst",
        nargs="?",
        help="output .safetensors; if omitted, derived from SRC by "
        "replacing bf16/fp16/fp32 with int8_convrot (or appending _int8_convrot)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument(
        "--w4a8",
        action="store_true",
        help="quantize eligible block linears as W4A8 (int4 weight + codebook "
        "+ fp8 scales, int8 activations) instead of int8. Layers with K not divisible by "
        "256 or N<64 fall back to int8; embeddings stay int8.",
    )
    fmt.add_argument(
        "--w6a8",
        action="store_true",
        help="like --w4a8 but uniform 6-bit weights: ~3x lower weight error than "
        "W4A8 at 0.78 vs 0.56 bytes/weight, same speed",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="columns per fp8 group scale for --w4a8/--w6a8 (default 16 for W4A8, 32 for W6A8)",
    )
    ap.add_argument("--exclude", default=None, help="regex; matching layers are FORCED to passthrough")
    ap.add_argument("--include", default=None, help="regex; matching eligible layers are FORCED to quantize")
    ap.add_argument(
        "--min-gemm",
        type=int,
        default=256,
        help="skip a layer if min(N,K) < this (default 256: a GEMM whose small side is "
        "under ~256 never beats bf16 at any M, so int8 is pure overhead). --min-gemm 0 disables.",
    )
    ap.add_argument(
        "--quant-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="quantize token-embedding tables (embed_tokens, embed_tokens_per_layer, ...) "
        "per-ROW int8 (+rotation); ON by default (often the biggest size win). "
        "--no-quant-embeddings keeps them bf16.",
    )
    ap.add_argument(
        "--mseclip",
        action="store_true",
        help="MSE-optimal clip instead of absmax for the CONVROT linears only (embeddings always absmax) (~2-3%% lower weight error, but a proxy — validate output before trusting it)",
    )
    ap.add_argument(
        "--downcast-fp32", action="store_true", help="downcast stray fp32 passthrough linears to compute dtype"
    )
    ap.add_argument(
        "--warn-thresh",
        type=float,
        default=None,
        help="warn on any quantized layer whose relerr%% exceeds this (default 2.0 for int8, "
        "4.0 for W6A8, 10.0 for W4A8: their normal weight error is ~2.4%% / ~7%%)",
    )
    ap.add_argument(
        "--verify-report", default=None, help="write the full per-layer (relerr, cos, gs) table to this path"
    )
    args = ap.parse_args()
    bits = 6 if args.w6a8 else 4
    args.w4a8 = args.w4a8 or args.w6a8  # one grouped-int path below, parametrized by bits
    if args.warn_thresh is None:
        args.warn_thresh = {4: 10.0, 6: 4.0}[bits] if args.w4a8 else 2.0
    if not args.dst and not args.dry_run:
        # derive dst from src: swap dtype token for int8_convrot (else append), always .safetensors
        base = os.path.splitext(os.path.basename(args.src))[0]
        tag = f"w{bits}a8_convrot" if args.w4a8 else "int8_convrot"
        new = re.sub(r"(?i)(bf16|fp16|fp32)", tag, base)
        if new == base:
            new = base + "_" + tag
        args.dst = os.path.join(os.path.dirname(args.src), new + ".safetensors")
        print(f"auto dst -> {args.dst}")
    exc = re.compile(args.exclude) if args.exclude else None
    inc = re.compile(args.include) if args.include else None

    with open_model(args.src) as st:  # .safetensors (lazy) or .pth/.pt/.ckpt (safe load)
        src_meta = st.metadata() or {}
        keys = list(st.keys())
        scaled = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}  # fp8 sources
        # compute/passthrough dtype = dominant non-fp8 float weight dtype
        dtc = collections.Counter(st.get_slice(k).get_dtype() for k in keys if k.endswith(".weight"))
        target = torch.float16 if dtc.get("F16", 0) >= dtc.get("BF16", 0) and dtc.get("F16", 0) else torch.bfloat16

        # ---- plan ----
        plan = []  # (base, shape, gs)  block linears -> int8+convrot
        eplan = []  # (base, shape)      token embeddings -> per-row int8
        skip = collections.Counter()
        for key in keys:
            if not key.endswith(".weight"):
                continue
            base = key[: -len(".weight")]
            shape = tuple(st.get_slice(key).get_shape())
            if is_token_embedding(base, shape):
                if args.quant_embeddings and not (exc and exc.search(base)):
                    eplan.append((base, shape))
                else:
                    skip["embedding(--no-quant-embeddings/excluded)"] += 1
                continue
            q, reason = classify(base, shape)
            if exc and exc.search(base):
                q, reason = False, "excluded(flag)"
            if inc and inc.search(base) and len(shape) == 2 and best_gs(shape[1]) and shape[0] >= 8:
                q, reason = True, f"gs{best_gs(shape[1])}(incl-flag)"
            if q and args.min_gemm and min(shape) < args.min_gemm:
                q, reason = False, f"below-min-gemm({min(shape)})"
            if q:
                plan.append((base, shape, best_gs(shape[1])))
            else:
                skip[reason] += 1

        # ---- report ----
        by_pat = collections.defaultdict(lambda: [0, None, None])
        qparams = 0
        for base, shape, gs in plan:
            pat = re.sub(r"\d+", "N", base)
            by_pat[pat][0] += 1
            by_pat[pat][1] = shape
            by_pat[pat][2] = gs
            qparams += shape[0] * shape[1]
        print(f"SRC {args.src}")
        print(f"compute/passthrough dtype: {target}")
        fmt_label = (
            f"W{bits}A8+convrot (int8 fallback for K%256!=0 or N<64)"
            if args.w4a8
            else f"int8+convrot, {'MSE-clip' if args.mseclip else 'absmax'}"
        )
        print(f"\nQUANTIZE {len(plan)} layers ({fmt_label}):")
        for pat in sorted(by_pat):
            c, shape, gs = by_pat[pat]
            print(f"  x{c:<4d} gs{gs:<3d} {str(shape):16s} {pat}")
        gsdist = collections.Counter(gs for _, _, gs in plan)
        print(f"  groupsizes: {dict(gsdist)}   quantized params: {qparams / 1e9:.2f}B  (~{qparams / 1e9:.1f} GB int8)")
        if eplan:
            ep = sum(s[0] * s[1] for _, s in eplan)
            print(f"\nEMBEDDINGS {len(eplan)} table(s) per-row int8 (+rotation):")
            for b, s in eplan:
                print(f"  {str(s):20s} {b}")
            print(f"  embedding params: {ep / 1e9:.2f}B  (~{ep / 1e9:.1f} GB int8, saves ~{ep / 1e9:.1f} GB vs bf16)")
        print(f"\nLEAVE AS-IS ({sum(skip.values())} weights):")
        for reason, c in skip.most_common():
            print(f"  x{c:<4d} {reason}")
        if args.dry_run:
            print("\n[dry-run] nothing written.")
            return

        # ---- execute ----
        quant_set = {b for b, _, _ in plan}
        embed_set = {b for b, _ in eplan}
        out = {}
        nq = 0
        t0 = time.time()
        errs = []  # (relerr%, cos, gs, base) per quantized layer
        for key in keys:
            if key.endswith(".weight_scale"):
                continue  # fp8 source scale: consumed by dequant
            t = st.get_tensor(key)
            if not key.endswith(".weight"):
                out[key] = t
                continue
            base = key[: -len(".weight")]
            # materialize source weight (dequant fp8 rowwise if needed)
            if t.dtype in FP8 and base in scaled:
                sc = st.get_tensor(base + ".weight_scale").float()
                w = t.float() * sc.view(-1, 1)
            elif t.dtype in FP8:
                w = t.float()
            else:
                w = t
            if base in embed_set:
                egs = best_gs(w.shape[1])  # rotate if the row dim allows it
                qd, scale = quantize_embedding(w, egs)
                # relerr, chunked with f64 accumulation (a single fp32 .norm() over a ~1B-element
                # vocab table accumulates enough error to misreport by >0.1pp)
                h_e = _build_hadamard(egs, device="cuda", dtype=torch.float32) if egs else None
                se = sw = 0.0
                for i in range(0, w.shape[0], 32768):
                    wf = w[i : i + 32768].to("cuda", torch.float32)
                    dq = qd[i : i + 32768].to("cuda").float() * scale[i : i + 32768].to("cuda")
                    if h_e is not None:
                        dq = _rotate_weight(dq, h_e, egs)  # un-rotate to compare against source
                    se += ((dq - wf) ** 2).sum().double().item()
                    sw += (wf**2).sum().double().item()
                    del wf, dq
                torch.cuda.empty_cache()
                relerr = (se / sw) ** 0.5 * 100.0
                rot = f"rotated gs{egs} + " if egs else ""
                print(f"  embedding {base} {tuple(w.shape)} {rot}per-row int8 relerr={relerr:.3f}%", flush=True)
                ecfg = {"format": "int8_tensorwise"}
                if egs is not None:
                    ecfg["convrot"] = True
                    ecfg["convrot_groupsize"] = egs
                out[key] = qd
                out[f"{base}.weight_scale"] = scale
                out[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(ecfg).encode("utf-8")), dtype=torch.uint8)
                torch.cuda.empty_cache()
                continue
            if base in quant_set and args.w4a8 and best_gs(w.shape[1]) == 256 and w.shape[0] >= 64:
                tensors, cfg, relerr = quantize_w4a8(w, bits, args.group_size)
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} W{bits}A8 relerr={relerr:.2f}%", flush=True)
                errs.append((relerr, 1.0, 256, base))
                for suf, val in tensors.items():
                    out[f"{base}.{suf}"] = val
                out[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} W{bits}A8 relerr={relerr:.2f}%", flush=True)
            elif base in quant_set:
                gs = best_gs(w.shape[1])
                qd, scale = quantize_convrot(w, gs, mseclip=args.mseclip)
                cos, relerr = recon_metrics(qd, scale, w, gs)
                assert cos > 0.99, f"BROKEN quant (rotation/format?) {base} cos={cos:.5f} relerr={relerr:.2f}%"
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
                errs.append((relerr, cos, gs, base))
                out[key] = qd.cpu()
                out[f"{base}.weight_scale"] = scale.cpu()
                out[f"{base}.comfy_quant"] = cq_tensor(gs)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
            else:
                # passthrough: fp8 must be de-fp8'd; fp32 optionally downcast; else keep source dtype
                if t.dtype in FP8:
                    out[key] = w.to(target)
                elif (
                    t.dtype == torch.float32
                    and args.downcast_fp32
                    and not (base.endswith(".scale") or EXCLUDE_SEG.search(base.split(".")[-1]))
                ):
                    out[key] = w.to(target)
                else:
                    out[key] = t
            torch.cuda.empty_cache()
        save_file(out, args.dst, metadata=dict(src_meta))
        print(f"DONE: quantized {nq} layers, {len(out)} tensors, {time.time() - t0:.1f}s -> {args.dst}")

        # ---- per-layer error report ----
        if errs:
            errs.sort(reverse=True)  # worst relerr first
            rvals = [e[0] for e in errs]
            mean = sum(rvals) / len(rvals)
            over = [e for e in errs if e[0] > args.warn_thresh]
            per_gs = collections.defaultdict(list)
            for r, c, gs, b in errs:
                per_gs[gs].append(r)
            print("\n=== quant error (relerr = ||dequant-source|| / ||source||) ===")
            print(f"  mean {mean:.3f}%   min {min(rvals):.3f}%   max {max(rvals):.3f}%   layers {len(errs)}")
            print(
                "  per groupsize: "
                + "  ".join(
                    f"gs{gs}: mean {sum(v) / len(v):.3f}% max {max(v):.3f}% (x{len(v)})"
                    for gs, v in sorted(per_gs.items())
                )
            )
            print("  worst 8 layers:")
            for r, c, gs, b in errs[:8]:
                print(f"    {r:6.3f}%  cos {c:.5f}  gs{gs:<3d} {b}")
            if over:
                print(f"  !! {len(over)} layer(s) over --warn-thresh ({args.warn_thresh}%) — review above")
        if args.verify_report and errs:
            with open(args.verify_report, "w") as f:
                f.write("relerr_pct\tcosine\tgroupsize\tlayer\n")
                for r, c, gs, b in errs:
                    f.write(f"{r:.4f}\t{c:.6f}\t{gs}\t{b}\n")
            print(f"  full per-layer table -> {args.verify_report}")


if __name__ == "__main__":
    main()
