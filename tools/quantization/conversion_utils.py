"""Shared helpers for native SenseNova-U1 artifact conversion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
MANIFEST_SCHEMA = "sensenova-artifact-manifest-v1"


def prepare_output(source: str | Path, output: str | Path) -> Path:
    """Validate conversion resources and reserve an output partial file."""
    source, output = Path(source), Path(output)
    for name in CONFIG_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_suffix(output.suffix + ".partial")


def tensors(source: str | Path):
    """Read one tensor at a time from an HF directory or merged Safetensors file."""
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


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON atomically so an interrupted conversion leaves the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_quantization_config(output: str | Path, config: dict[str, Any]) -> Path:
    """Write an artifact-specific quantization configuration beside its weights."""
    output = Path(output)
    path = output.with_suffix(".quantization_config.json")
    write_json(path, config)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_directories(output_dir: Path) -> tuple[Path, ...]:
    """Return the artifact directory and the requested top-level ``sensenova`` directory."""
    directories = [output_dir]
    if output_dir.parent.name == "sensenova":
        directories.append(output_dir.parent)
    return tuple(directories)


def _artifact_entry(report_path: Path, report: dict[str, Any], base: Path) -> dict[str, Any]:
    artifact = Path(report["artifact"])
    artifact_path = report_path.parent / artifact.name
    entry = {
        "path": artifact.name
        if report_path.parent == base
        else str(report_path.parent.relative_to(base) / artifact.name),
        "conversion_report": str(report_path.relative_to(base)),
    }
    for key, value in report.items():
        if key not in {"artifact", "output", "source"}:
            entry[key] = value
    quantization_config = artifact_path.with_suffix(".quantization_config.json")
    if quantization_config.is_file():
        entry["quantization_config"] = str(quantization_config.relative_to(base))
    return entry


def _update_manifest(base: Path) -> None:
    reports = []
    report_paths = (
        sorted(base.glob("*/*.conversion.json")) if base.name == "sensenova" else sorted(base.glob("*.conversion.json"))
    )
    for report_path in report_paths:
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "artifact" in report:
            reports.append(_artifact_entry(report_path, report, base))
    write_json(
        base / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "artifacts": reports,
        },
    )


def finish(partial: str | Path, output: str | Path, source: str | Path, **details: Any) -> None:
    """Publish a verified artifact, copy resources, and update its manifests."""
    partial, output, source = Path(partial), Path(output), Path(source)
    digest = _sha256(partial)
    partial.rename(output)
    for name in CONFIG_FILES:
        shutil.copy2(source / name, output.parent / name)
    report = {
        "status": "passed",
        "artifact": output.name,
        "output": output.name,
        "source_model": source.name,
        "source_config_sha256": _sha256(source / "config.json"),
        "resources": list(CONFIG_FILES),
        "sha256": digest,
        **details,
    }
    report_path = output.with_suffix(".conversion.json")
    write_json(report_path, report)
    for directory in _manifest_directories(output.parent):
        _update_manifest(directory)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
