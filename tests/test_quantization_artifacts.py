import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "quantization"))

from conversion_utils import CONFIG_FILES, finish, write_quantization_config
from publish_hf import validate_release_tree


def _source_directory(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    for name in CONFIG_FILES:
        (source / name).write_text(name)
    return source


def _publish(
    root: Path, source: Path, directory: str, name: str, fmt: str, quantization_config: dict | None = None
) -> None:
    output_dir = root / "sensenova" / directory
    output = output_dir / name
    partial = output.with_suffix(output.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(name.encode())
    if quantization_config is not None:
        write_quantization_config(output, quantization_config)
    finish(partial, output, source, format=fmt, tensors=1)


def test_conversion_reports_and_manifests_are_artifact_specific(tmp_path: Path) -> None:
    source = _source_directory(tmp_path)
    _publish(tmp_path, source, "SenseNova-U1.6-8B-MoT-Native", "SenseNova-U1.6-BF16.safetensors", "bf16")
    _publish(
        tmp_path,
        source,
        "SenseNova-U1.6-8B-MoT-Native",
        "SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors",
        "fp8",
        {"format": "fp8"},
    )
    _publish(
        tmp_path,
        source,
        "SenseNova-U1.6-8B-MoT-GGUF",
        "SenseNova-U1.6-8B-MoT-Q6_K.gguf",
        "Q6_K",
    )
    output_dir = tmp_path / "sensenova" / "SenseNova-U1.6-8B-MoT-Native"
    assert (output_dir / "SenseNova-U1.6-BF16.conversion.json").is_file()
    assert (output_dir / "SenseNova-U1.6-FP8-E4M3FN-W8A8.conversion.json").is_file()

    local_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert [item["path"] for item in local_manifest["artifacts"]] == [
        "SenseNova-U1.6-BF16.safetensors",
        "SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors",
    ]
    assert local_manifest["artifacts"][1]["quantization_config"] == (
        "SenseNova-U1.6-FP8-E4M3FN-W8A8.quantization_config.json"
    )

    root_manifest = json.loads((tmp_path / "sensenova" / "manifest.json").read_text())
    assert [item["path"] for item in root_manifest["artifacts"]] == [
        "SenseNova-U1.6-8B-MoT-GGUF/SenseNova-U1.6-8B-MoT-Q6_K.gguf",
        "SenseNova-U1.6-8B-MoT-Native/SenseNova-U1.6-BF16.safetensors",
        "SenseNova-U1.6-8B-MoT-Native/SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors",
    ]
    validation = validate_release_tree(tmp_path / "sensenova")
    assert validation["artifact_count"] == 3
    assert "manifest.json" in validation["upload_paths"]


def test_quantization_config_is_named_after_its_artifact(tmp_path: Path) -> None:
    first = tmp_path / "SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors"
    second = tmp_path / "SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors"
    write_quantization_config(first, {"format": "fp8"})
    write_quantization_config(second, {"format": "int8-convrot"})

    assert json.loads(first.with_suffix(".quantization_config.json").read_text()) == {"format": "fp8"}
    assert json.loads(second.with_suffix(".quantization_config.json").read_text()) == {"format": "int8-convrot"}
