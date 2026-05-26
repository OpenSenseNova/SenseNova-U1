# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""Convert internevo checkpoints to HuggingFace safetensors format.

Handles both dense and MoE (incl. MoT U/G dual-branch) models. Reads internevo
shards, merges along the weight-parallel dimension, splits MoE experts where
present, renames keys via ``key_mappings.json``, and writes a sharded
HuggingFace directory:

* ``model-{NNNNN}-of-{16}.safetensors``           — dense (vit, mlp1, fm_modules, llm dense layers, MoE gates)
* ``moemodel-{NNNNN}-of-{N}.safetensors``         — MoE experts, one file per LLM MoE layer (contains both U and G branches when MoT)
* ``model.safetensors.index.json``                — combined index covering everything

Memory: shards are loaded with ``torch.load(mmap=True)`` and processed one
output slice at a time, so peak RAM tracks the largest slice rather than the
total checkpoint size. Works inside containers with ~32GB cgroup limits even
for >100GB checkpoints.

Supported layouts (same scope as the training stack writes):

* Tensor parallel (TP):       1 only
* Pipeline parallel (PP):     1 only
* Weight  parallel (WP):      any N
* Expert  parallel (EP):      any N         (MoE only)
* MoT U/G expert files:       'interleaved' (file_L = 2*L / 2*L+1)
                              or 'offset'   (file_L = L / L+num_layers)
                              — auto-detected.

If ``model_config.pt`` was pickled inside an environment that had
``internlm``/``flash_attn`` and they're not available here, the loader
installs benign stubs so unpickling can proceed.

Usage::

    # Dense or MoE — same command; layout is auto-detected.
    python tools/revert2hf.py \\
        --src /path/to/RUN/<job>/<step> \\
        --tgt /path/to/output/hf_dir \\
        --extras-from /path/to/source-hf-model    # optional: copy tokenizer + config.json
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import types

import torch
from safetensors.torch import save_file
from tqdm import tqdm


# Keys whose tensors are replicated (not split) across WP shards.
NON_SPLIT_KEYS = [
    "fm_modules",
    "language_model.norm",
    "moe_layer.gate",
    "vision_model.embeddings",
    "attn.k_norm",
    "attn.q_norm",
    "ls1",
    "ls2",
    "norm1",
    "norm2",
    "norm1.bias",
    "norm2.bias",
    "mlp1.0.weight",
    "mlp1.0.bias",
    "e_score_correction_bias",
]

# Keys whose tensors are split along dim=1 (rather than the default dim=0).
DIM1_KEYS = ["tok_embeddings"]

# Final rename applied after the per-key mapping table, to align with HF's
# LlamaForCausalLM-style layout where transformer layers live under `model.`.
LLM_GLOBAL_KEY_MAPPING = ("language_model.layers", "language_model.model.layers")

# MoE-specific renames. Two-pass: wrapper (U or G branch), then inner w1/w2/w3.
MOE_WRAPPER_MAPS = [
    ("feed_forward.moe_layer.experts.wrapped_experts", "mlp.experts"),
    ("feed_forward_mot_gen.moe_layer.experts.wrapped_experts", "mlp_mot_gen.experts"),
]
MOE_INNER_MAP = [
    ("w1.weight", "gate_proj.weight"),
    ("w3.weight", "up_proj.weight"),
    ("w2.weight", "down_proj.weight"),
]

# Files copied verbatim from --extras-from to make the output a self-contained
# HF directory (architecture/tokenizer don't change across training steps).
EXTRA_FILES = [
    "config.json",
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]

_LAYER_RE = re.compile(r"\.(\d+)\.")
_LAYER_FOR_REMAP_RE = re.compile(r"(layers\.)(\d+)(\.)")
_WP_FILE_RE = re.compile(r"model_wp(\d+)_pp(\d+)\.pt")
_TP_FILE_RE = re.compile(r"model_tp\d+_pp\d+\.pt")
_MOE_FILE_RE = re.compile(r"model_moe_layer(\d+)_expert(\d+)_wp(\d+)\.pt")
_EXPERT_IDX_RE = re.compile(r"wrapped_experts\.\d+\.w")


# ---------------------------------------------------------------------------
# Pickle compatibility: stub out missing modules so torch.load can hydrate
# model_config.pt even when the writer's environment isn't fully present.
# ---------------------------------------------------------------------------

class _DictStub(dict):
    """A dict subclass that also accepts attribute-style access. Used as a stand-in
    for config classes (e.g. internlm Config) that pickle expects to instantiate."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class _StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        cls = type(name, (_DictStub,), {})
        setattr(self, name, cls)
        return cls


class _StubFinder:
    """Install stub modules on demand for known-missing pickle dependencies."""

    STUB_PREFIXES = ("internlm", "flash_attn")

    def find_module(self, name, path=None):
        if name.startswith(self.STUB_PREFIXES):
            return self

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        m = _StubModule(name)
        sys.modules[name] = m
        return m


def _install_pickle_stubs() -> None:
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


# ---------------------------------------------------------------------------
# Key rename helpers
# ---------------------------------------------------------------------------

def load_mapping(typ: str) -> list:
    """Load the (intern -> HF) key rename table for a model type."""
    path = os.path.join(os.path.dirname(__file__), "key_mappings.json")
    with open(path) as f:
        table = json.load(f)
    if typ not in table:
        raise ValueError(f"Unknown --typ {typ!r}. Available: {sorted(table)}")
    return table[typ]


def convert_dense_key(old_key: str, mapping: list) -> str:
    """Rename a non-expert key. Applies the first matching mapping rule, then
    the global ``language_model.layers`` -> ``language_model.model.layers`` fixup."""
    stripped = old_key[6:] if old_key.startswith("model.") else old_key
    new_key = stripped
    for intern_k, hf_k in mapping:
        if intern_k in stripped:
            new_key = stripped.replace(intern_k, hf_k)
            break
    if LLM_GLOBAL_KEY_MAPPING[0] in new_key:
        new_key = new_key.replace(*LLM_GLOBAL_KEY_MAPPING)
    return new_key


def convert_expert_key(stripped_key: str) -> str:
    """Rename an MoE expert key: ``wrapped_experts`` -> ``experts`` (handles
    both U and G/`_mot_gen` branches) then the inner ``w1/w2/w3``."""
    new_key = stripped_key
    for intern_prefix, hf_prefix in MOE_WRAPPER_MAPS:
        if intern_prefix in new_key:
            new_key = new_key.replace(intern_prefix, hf_prefix)
            break
    for intern_k, hf_k in MOE_INNER_MAP:
        if intern_k in new_key:
            new_key = new_key.replace(intern_k, hf_k)
            break
    if LLM_GLOBAL_KEY_MAPPING[0] in new_key:
        new_key = new_key.replace(*LLM_GLOBAL_KEY_MAPPING)
    return new_key


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def merge_wp_shards(shards: list, key: str) -> torch.Tensor:
    """Concatenate (or replicate, for non-split keys) one key across WP shards."""
    if any(nsk in key for nsk in NON_SPLIT_KEYS):
        return shards[0][key].clone()
    dim = 1 if any(d1k in key for d1k in DIM1_KEYS) else 0
    return torch.cat([s[key] for s in shards], dim=dim)


def maybe_reduce_norm(key: str, merged: torch.Tensor, num_wp: int) -> torch.Tensor:
    """Norms are wp-replicated but stored full-size in some configurations;
    take the first chunk to recover the true norm weight."""
    if "_norm" in key and "fm_modules" not in key:
        return merged.chunk(num_wp, dim=-1)[0].contiguous()
    return merged


def get_layer_idx(key: str):
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def remap_layer_idx_in_key(k: str, logical_layer_idx: int, logical_num_layers: int) -> str:
    """Force a key's transformer-layer index to ``logical_layer_idx``. Mirrors the
    safety remap from the reference impl: some checkpoints store an out-of-range
    or shifted layer id inside the tensor keys, which would otherwise drop them."""
    m = _LAYER_FOR_REMAP_RE.search(k)
    if not m:
        return k
    layer_idx = int(m.group(2))
    if layer_idx >= logical_num_layers or layer_idx != logical_layer_idx:
        return k[: m.start()] + f"{m.group(1)}{logical_layer_idx}{m.group(3)}" + k[m.end():]
    return k


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------

def discover_shards(src: str, num_layers: int, first_k_dense_replace: int):
    """Detect the checkpoint layout in ``src``.

    Returns:
        wp_files:   sorted list of dense WP shard filenames.
        moe_meta:   ``None`` if no MoE files exist, else dict with:
                    ``ep_size``       — number of expert-parallel groups
                    ``ewp_size``      — number of WP shards within each ep group
                    ``layout``        — 'interleaved' (file_L = 2*L / 2*L+1)
                                        or 'offset'  (file_L = L / L+num_layers)
    """
    files = os.listdir(src)

    tp_files = [f for f in files if _TP_FILE_RE.fullmatch(f)]
    if tp_files:
        raise NotImplementedError(
            f"TP > 1 layout (model_tp*_pp*.pt) is not supported; found: {tp_files[:3]}..."
        )

    wp_files = sorted(
        [f for f in files if _WP_FILE_RE.fullmatch(f)],
        key=lambda f: int(_WP_FILE_RE.match(f).group(1)),
    )
    if not wp_files:
        raise FileNotFoundError(f"No model_wp*_pp0.pt shards found in {src}")
    pp_indices = {int(_WP_FILE_RE.match(f).group(2)) for f in wp_files}
    if pp_indices != {0}:
        raise NotImplementedError(
            f"PP > 1 layout is not supported; found pp indices: {sorted(pp_indices)}"
        )

    moe_files = [f for f in files if _MOE_FILE_RE.fullmatch(f)]
    if not moe_files:
        return wp_files, None

    eps, wps = set(), set()
    file_layers = set()
    for f in moe_files:
        m = _MOE_FILE_RE.match(f)
        file_layers.add(int(m.group(1)))
        eps.add(int(m.group(2)))
        wps.add(int(m.group(3)))
    ep_size = len(eps)
    ewp_size = len(wps)

    # Auto-detect U/G file naming layout from the first MoE-eligible layer.
    layout = "interleaved"
    if first_k_dense_replace < num_layers:
        probe_L = first_k_dense_replace
        if os.path.exists(os.path.join(src, f"model_moe_layer{2 * probe_L + 1}_expert0_wp0.pt")):
            layout = "interleaved"
        elif os.path.exists(
            os.path.join(src, f"model_moe_layer{probe_L + num_layers}_expert0_wp0.pt")
        ):
            layout = "offset"
        else:
            # No G-branch detected → either a pure dense-MoE model (no MoT), or
            # the G files genuinely aren't there. Default to interleaved so U-only
            # processing still works.
            layout = "interleaved"

    return wp_files, {
        "ep_size": ep_size,
        "ewp_size": ewp_size,
        "layout": layout,
    }


# ---------------------------------------------------------------------------
# Slice classification (dense path)
# ---------------------------------------------------------------------------

def classify_dense_key(stripped_key: str, vit_layeridxs: list, llm_layeridxs: list):
    """Return ``(category, slice_idx)`` for a dense (non-expert) key, or ``None``
    if the key doesn't belong to any known category (and will be skipped)."""
    if "vision_model" in stripped_key:
        lid = get_layer_idx(stripped_key)
        if lid is None:
            return ("vit", 0)
        for i, layers in enumerate(vit_layeridxs):
            if lid in layers:
                return ("vit", i)
        return None
    if "mlp1" in stripped_key:
        return ("mlp1", 0)
    if "fm_modules" in stripped_key:
        return ("fm_modules", 0)
    if "language_model" in stripped_key:
        lid = get_layer_idx(stripped_key)
        if lid is None:
            return ("llm", 0)
        for i, layers in enumerate(llm_layeridxs):
            if lid in layers:
                return ("llm", i)
        return None
    return None


def compute_dense_slice_layout(vit_num_layers: int, llm_num_layers: int):
    """16-slice dense layout matching the publicly-released checkpoints.

    Returns (vit_layeridxs, llm_layeridxs, total_dense_slices).
    """
    vit_slices = 3
    num_vit_per_slice = vit_num_layers // vit_slices if vit_slices else 0
    if vit_num_layers > 0 and vit_num_layers % vit_slices > 0:
        vit_slices += 1
    vit_layeridxs = []
    for i in range(vit_slices):
        if i < 3:
            vit_layeridxs.append(list(range(i * num_vit_per_slice, (i + 1) * num_vit_per_slice)))
        else:
            vit_layeridxs.append(list(range(i * num_vit_per_slice, vit_num_layers)))

    llm_slices = 10
    num_llm_per_slice = llm_num_layers // llm_slices
    if llm_num_layers % llm_slices > 0:
        llm_slices += 1
    llm_layeridxs = []
    for i in range(llm_slices):
        if i < 10:
            llm_layeridxs.append(list(range(i * num_llm_per_slice, (i + 1) * num_llm_per_slice)))
        else:
            llm_layeridxs.append(list(range(i * num_llm_per_slice, llm_num_layers)))

    total_dense_slices = vit_slices + 1 + 1 + llm_slices  # vit + mlp1 + fm_modules + llm
    return vit_layeridxs, llm_layeridxs, total_dense_slices


def build_dense_slot_index(vit_slices: int, llm_slices: int) -> dict:
    """Map ``(category, slice_idx)`` to 1-indexed global dense slice number."""
    slot_to_global = {}
    g = 0
    for i in range(vit_slices):
        g += 1
        slot_to_global[("vit", i)] = g
    g += 1
    slot_to_global[("mlp1", 0)] = g
    g += 1
    slot_to_global[("fm_modules", 0)] = g
    for i in range(llm_slices):
        g += 1
        slot_to_global[("llm", i)] = g
    return slot_to_global


# ---------------------------------------------------------------------------
# MoE per-layer processing
# ---------------------------------------------------------------------------

def _file_L_for(layout: str, actual_L: int, num_layers: int, is_gen: bool) -> int:
    if layout == "interleaved":
        return 2 * actual_L + (1 if is_gen else 0)
    elif layout == "offset":
        return actual_L + (num_layers if is_gen else 0)
    raise ValueError(f"unknown MoE layout: {layout}")


def process_moe_layer(src: str, actual_L: int, num_layers: int,
                     ep_size: int, ewp_size: int,
                     lm_num_experts: int, gen_num_experts: int,
                     layout: str) -> dict:
    """For one actual LLM layer, load U and G branch MoE files, split into
    per-expert chunks, apply rename, return ``{hf_key: tensor}``.

    Memory: per (branch, ep_id), WP shards are mmap'd and merged key-by-key
    then split into per-expert chunks. Peak RAM ~= one layer's expert weights.
    """
    assert lm_num_experts % ep_size == 0, (
        f"lm_num_experts={lm_num_experts} not divisible by ep_size={ep_size}"
    )
    assert gen_num_experts % ep_size == 0, (
        f"gen_num_experts={gen_num_experts} not divisible by ep_size={ep_size}"
    )
    lm_per_ep = lm_num_experts // ep_size
    gen_per_ep = gen_num_experts // ep_size

    out = {}
    for is_gen in (False, True):
        file_L = _file_L_for(layout, actual_L, num_layers, is_gen)
        per_ep = gen_per_ep if is_gen else lm_per_ep

        # Skip silently if this branch has no files (e.g., MoE-only / no-MoT model).
        if not os.path.exists(
            os.path.join(src, f"model_moe_layer{file_L}_expert0_wp0.pt")
        ):
            continue

        for ep_id in range(ep_size):
            shards = []
            for wp_id in range(ewp_size):
                f = os.path.join(
                    src, f"model_moe_layer{file_L}_expert{ep_id}_wp{wp_id}.pt"
                )
                shards.append(
                    torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                )

            for k in list(shards[0].keys()):
                merged = merge_wp_shards(shards, k)
                # Safety remap: force layer idx in the key to actual_L.
                k_fixed = remap_layer_idx_in_key(k, actual_L, num_layers)
                expert_chunks = merged.chunk(per_ep, dim=0)
                for local_eid in range(per_ep):
                    global_eid = ep_id * per_ep + local_eid
                    rewired = _EXPERT_IDX_RE.sub(
                        f"wrapped_experts.{global_eid}.w", k_fixed
                    )
                    stripped = rewired[6:] if rewired.startswith("model.") else rewired
                    hf_key = convert_expert_key(stripped)
                    # Transpose to align with HF's (in_features, out_features) layout.
                    out[hf_key] = (
                        expert_chunks[local_eid].transpose(-1, -2).contiguous().clone()
                    )

            del shards
            gc.collect()

    return out


# ---------------------------------------------------------------------------
# Extras copy
# ---------------------------------------------------------------------------

def copy_extras(extras_from: str, tgt: str) -> None:
    print(f"\nCopying extras from {extras_from} ...")
    for name in EXTRA_FILES:
        src_path = os.path.join(extras_from, name)
        if not os.path.isfile(src_path):
            print(f"  skip (missing): {name}")
            continue
        shutil.copy2(src_path, os.path.join(tgt, name))
        print(f"  copied: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(src: str, tgt: str, typ: str, extras_from: str | None = None) -> None:
    os.makedirs(tgt, exist_ok=True)
    mapping = load_mapping(typ)
    t0 = time.time()

    print(f"[1/5] Loading model_config.pt from {src} ...")
    _install_pickle_stubs()
    model_config = torch.load(
        os.path.join(src, "model_config.pt"), map_location="cpu", weights_only=False
    )
    vit_num_layers = model_config["vit_cfg"]["num_hidden_layers"]
    llm_num_layers = model_config["num_layers"]
    moe_kwargs = model_config.get("moe_kwargs") or {}
    first_k_dense_replace = int(moe_kwargs.get("first_k_dense_replace", 0))
    lm_num_experts = int(moe_kwargs.get("num_experts", 0))
    gen_num_experts = int(moe_kwargs.get("gen_num_experts", lm_num_experts))
    print(f"      vit_num_layers={vit_num_layers}, llm_num_layers={llm_num_layers}")
    print(
        f"      moe: lm_num_experts={lm_num_experts}, gen_num_experts={gen_num_experts}, "
        f"first_k_dense_replace={first_k_dense_replace}"
    )

    vit_layeridxs, llm_layeridxs, total_dense_slices = compute_dense_slice_layout(
        vit_num_layers, llm_num_layers
    )
    vit_slices = len(vit_layeridxs)
    llm_slices = len(llm_layeridxs)
    slot_to_global = build_dense_slot_index(vit_slices, llm_slices)
    print(
        f"      dense slices: {vit_slices} vit + 1 mlp1 + 1 fm_modules + {llm_slices} llm "
        f"= {total_dense_slices}"
    )

    print(f"[2/5] Discovering shards ...")
    wp_files, moe_meta = discover_shards(src, llm_num_layers, first_k_dense_replace)
    num_wp = len(wp_files)
    print(f"      dense WP shards: {num_wp}")
    moe_layers = []
    if moe_meta is not None and lm_num_experts > 0:
        moe_layers = list(range(first_k_dense_replace, llm_num_layers))
        print(
            f"      MoE: {len(moe_layers)} layers, ep_size={moe_meta['ep_size']}, "
            f"ewp_size={moe_meta['ewp_size']}, layout={moe_meta['layout']}"
        )
    elif moe_meta is not None:
        print(f"      MoE: shards present but lm_num_experts=0 in config; ignoring MoE.")
        moe_meta = None
    else:
        print(f"      MoE: none detected")

    total_moe_files = len(moe_layers)
    total_files = total_dense_slices + total_moe_files
    print(f"      output: {total_dense_slices} dense + {total_moe_files} moemodel = {total_files} files")

    print(f"[3/5] Mmap-loading {num_wp} dense WP shards (lazy) ...")
    shards = []
    for fname in tqdm(wp_files):
        shards.append(
            torch.load(os.path.join(src, fname), map_location="cpu", mmap=True, weights_only=False)
        )
    all_keys = list(shards[0].keys())
    print(f"      dense source keys: {len(all_keys)}")

    # Group dense (non-expert) keys by destination slice + precompute renamed keys.
    slice_keys: dict[int, list[tuple[str, str]]] = {}
    unclassified = []
    for k in all_keys:
        stripped = k[6:] if k.startswith("model.") else k
        slot = classify_dense_key(stripped, vit_layeridxs, llm_layeridxs)
        if slot is None:
            unclassified.append(k)
            continue
        slice_keys.setdefault(slot_to_global[slot], []).append(
            (k, convert_dense_key(k, mapping))
        )
    if unclassified:
        print(f"      WARNING: {len(unclassified)} unclassified dense keys, skipping:")
        for k in unclassified[:10]:
            print(f"        {k}")

    print(f"[4/5] Writing dense slices ...")
    index_dict = {}
    for global_slot in range(1, total_dense_slices + 1):
        pairs = slice_keys.get(global_slot, [])
        slice_name = f"model-{global_slot:05d}-of-{total_dense_slices:05d}.safetensors"
        slice_path = os.path.join(tgt, slice_name)

        cur_states = {}
        if pairs:
            desc = f"  slice {global_slot:2d}/{total_dense_slices} dense ({len(pairs):3d} keys)"
            for orig_k, new_k in tqdm(pairs, desc=desc):
                merged = merge_wp_shards(shards, orig_k)
                merged = maybe_reduce_norm(orig_k, merged, num_wp)
                cur_states[new_k] = merged.contiguous().clone()
                index_dict[new_k] = slice_name
        save_file(cur_states, slice_path)
        size_mb = os.path.getsize(slice_path) / (1024 * 1024)
        print(f"      wrote {slice_name}  ({len(cur_states)} keys, {size_mb:.1f} MB)")
        del cur_states
        gc.collect()

    # Done with dense shards; release mmap before MoE processing.
    del shards
    gc.collect()

    if moe_layers:
        print(f"[5/5] Writing {total_moe_files} moemodel files (one per MoE layer) ...")
        for i, actual_L in enumerate(moe_layers, start=1):
            slice_name = f"moemodel-{i:05d}-of-{total_moe_files:05d}.safetensors"
            slice_path = os.path.join(tgt, slice_name)

            expert_states = process_moe_layer(
                src, actual_L,
                num_layers=llm_num_layers,
                ep_size=moe_meta["ep_size"],
                ewp_size=moe_meta["ewp_size"],
                lm_num_experts=lm_num_experts,
                gen_num_experts=gen_num_experts,
                layout=moe_meta["layout"],
            )
            for hf_key in expert_states:
                index_dict[hf_key] = slice_name

            save_file(expert_states, slice_path)
            size_mb = os.path.getsize(slice_path) / (1024 * 1024)
            print(
                f"      wrote {slice_name}  layer={actual_L}, "
                f"{len(expert_states)} expert keys, {size_mb:.1f} MB"
            )
            del expert_states
            gc.collect()
    else:
        print(f"[5/5] No MoE layers; skipped moemodel output.")

    index_path = os.path.join(tgt, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": index_dict}, f, indent=2)
    print(f"\nwrote {index_path}  ({len(index_dict)} entries)")

    if extras_from:
        copy_extras(extras_from, tgt)

    print(f"\nDone in {time.time() - t0:.1f}s. Output: {tgt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src", required=True,
        help="Input folder containing model_config.pt + model_wp{N}_pp0.pt "
             "(+ optional model_moe_layer{L}_expert{E}_wp{W}.pt for MoE).",
    )
    parser.add_argument(
        "--tgt", required=True,
        help="Output folder. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--typ", default="neo++_mot",
        help="Key-mapping schema name in tools/key_mappings.json. Default: %(default)s",
    )
    parser.add_argument(
        "--extras-from", default=None,
        help="Optional reference HF model dir to copy config.json + tokenizer files from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.src, args.tgt, args.typ, args.extras_from)
