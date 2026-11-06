from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 0.1
WORKER_TERMINATE_TIMEOUT = 5.0
_MODEL_ARTIFACT_SUFFIXES = {".gguf", ".safetensors", ".sft"}
_WORKER_PATH = Path(__file__).with_name("hf_download_worker.py")


def _has_complete_model_weights(snapshot: Path) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            shard_names = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            return False
        return bool(shard_names) and all((snapshot / shard_name).is_file() for shard_name in shard_names)
    return (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()


def _find_complete_cached_snapshot(repo_id: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None
    return str(snapshot) if _has_complete_model_weights(snapshot) else None


def _throw_if_comfyui_interrupted() -> None:
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()


def _terminate_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=WORKER_TERMINATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def download_hf_snapshot_interruptibly(repo_id: str, *, poll_interval: float = DEFAULT_POLL_INTERVAL) -> str:
    """Download a complete HF model snapshot while honoring ComfyUI interrupts."""
    if poll_interval < 0:
        raise ValueError("poll_interval must be >= 0.")

    cached_snapshot = _find_complete_cached_snapshot(repo_id)
    if cached_snapshot is not None:
        return cached_snapshot

    with tempfile.TemporaryDirectory(prefix="sensenova-hf-download-") as temp_dir:
        result_path = Path(temp_dir) / "result.json"
        command = [sys.executable, str(_WORKER_PATH), repo_id, str(result_path)]
        LOGGER.info("SenseNova U1 loader: downloading Hugging Face model %s", repo_id)
        process = subprocess.Popen(command)
        try:
            while process.poll() is None:
                _throw_if_comfyui_interrupted()
                time.sleep(poll_interval)
            _throw_if_comfyui_interrupted()
        except BaseException:
            _terminate_worker(process)
            raise

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Hugging Face snapshot worker exited with code {process.returncode} without a valid result."
            ) from exc

        if process.returncode != 0 or not payload.get("ok"):
            error_type = payload.get("error_type", "DownloadError")
            error = payload.get("error", f"worker exited with code {process.returncode}")
            raise RuntimeError(f"Hugging Face snapshot download failed for {repo_id!r}: {error_type}: {error}")

        snapshot_path = payload.get("snapshot_path")
        if not isinstance(snapshot_path, str) or not snapshot_path:
            raise RuntimeError(f"Hugging Face snapshot worker returned an invalid path for {repo_id!r}.")
        LOGGER.info("SenseNova U1 loader: Hugging Face model ready at %s", snapshot_path)
        return snapshot_path


def resolve_hf_model_snapshot_interruptibly(model_path: str) -> str:
    """Resolve an HF repo ID to a local snapshot; leave local artifacts unchanged."""
    if Path(model_path).exists() or Path(model_path).suffix.lower() in _MODEL_ARTIFACT_SUFFIXES:
        return model_path
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        validate_repo_id(model_path)
    except (ImportError, HFValidationError):
        return model_path
    return download_hf_snapshot_interruptibly(model_path)
