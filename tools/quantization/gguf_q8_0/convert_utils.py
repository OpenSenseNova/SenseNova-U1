"""Small file helpers. This copy belongs to this converter directory."""

import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open

CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def prepare_output(source, output):
    source, output = Path(source), Path(output)
    for name in CONFIG_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_suffix(output.suffix + ".partial")


def tensors(source):
    """Read one tensor at a time from an HF directory or merged Safetensors."""
    source = Path(source)
    index = None
    if source.is_dir():
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        files = [source / name for name in sorted(set(index.values()))]
    else:
        files = [source]
    seen = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in seen or (index is not None and index.get(name) != file.name):
                    raise ValueError(f"Duplicate or misindexed tensor: {name}")
                seen.add(name)
                yield name, reader.get_tensor(name)
    if index is not None and seen != set(index):
        raise ValueError("Missing tensors from source index")


def finish(partial, output, source, **details):
    output = Path(output)
    digest = hashlib.sha256()
    with partial.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    partial.rename(output)
    for name in CONFIG_FILES:
        shutil.copy2(Path(source) / name, output.parent / name)
    report = dict(status="passed", output=str(output), sha256=digest.hexdigest(), **details)
    (output.parent / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
