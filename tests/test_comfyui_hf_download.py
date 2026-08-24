import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_comfyui_source_path import LOCAL_PIPELINE

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DOWNLOAD_PATH = REPO_ROOT / "apps" / "comfyui" / "hf_download.py"


def _load_hf_download_module():
    module_name = "_sensenova_comfy_hf_download_test"
    spec = importlib.util.spec_from_file_location(module_name, HF_DOWNLOAD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {module_name: module}):
        spec.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class ComfyUIHFDownloadTest(unittest.TestCase):
    def test_terminate_worker_stops_a_real_child_process(self) -> None:
        hf_download = _load_hf_download_module()
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            hf_download._terminate_worker(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_interrupt_terminates_snapshot_worker_before_propagating(self) -> None:
        hf_download = _load_hf_download_module()
        process = _FakeProcess()

        class FakeInterrupt(BaseException):
            pass

        with (
            mock.patch.object(hf_download, "_find_complete_cached_snapshot", return_value=None),
            mock.patch.object(hf_download.subprocess, "Popen", return_value=process),
            mock.patch.object(hf_download, "_throw_if_comfyui_interrupted", side_effect=FakeInterrupt),
        ):
            with self.assertRaises(FakeInterrupt):
                hf_download.download_hf_snapshot_interruptibly("owner/model", poll_interval=0)

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_completed_snapshot_worker_returns_local_path(self) -> None:
        hf_download = _load_hf_download_module()

        def completed_process(command):
            Path(command[-1]).write_text(
                json.dumps({"ok": True, "snapshot_path": "/cache/models--owner--model/snapshots/revision"})
            )
            return _FakeProcess(returncode=0)

        with (
            mock.patch.object(hf_download, "_find_complete_cached_snapshot", return_value=None),
            mock.patch.object(hf_download.subprocess, "Popen", side_effect=completed_process),
            mock.patch.object(hf_download, "_throw_if_comfyui_interrupted"),
        ):
            snapshot = hf_download.download_hf_snapshot_interruptibly("owner/model", poll_interval=0)

        self.assertEqual(snapshot, "/cache/models--owner--model/snapshots/revision")

    def test_snapshot_worker_failure_is_reported_to_loader(self) -> None:
        hf_download = _load_hf_download_module()

        def failed_process(command):
            Path(command[-1]).write_text(
                json.dumps({"ok": False, "error_type": "RepositoryNotFoundError", "error": "not found"})
            )
            return _FakeProcess(returncode=1)

        with (
            mock.patch.object(hf_download, "_find_complete_cached_snapshot", return_value=None),
            mock.patch.object(hf_download.subprocess, "Popen", side_effect=failed_process),
            mock.patch.object(hf_download, "_throw_if_comfyui_interrupted"),
        ):
            with self.assertRaisesRegex(RuntimeError, "RepositoryNotFoundError: not found"):
                hf_download.download_hf_snapshot_interruptibly("owner/missing", poll_interval=0)

    def test_complete_cached_snapshot_skips_worker(self) -> None:
        hf_download = _load_hf_download_module()
        cached = "/cache/models--owner--model/snapshots/revision"

        with (
            mock.patch.object(hf_download, "_find_complete_cached_snapshot", return_value=cached),
            mock.patch.object(hf_download.subprocess, "Popen") as popen,
        ):
            snapshot = hf_download.download_hf_snapshot_interruptibly("owner/model")

        self.assertEqual(snapshot, cached)
        popen.assert_not_called()

    def test_local_model_loader_loads_downloaded_snapshot_but_keeps_repo_identity(self) -> None:
        load_model_and_tokenizer = mock.Mock(return_value=(object(), object()))
        fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        fake_core = SimpleNamespace(
            set_attn_backend=mock.Mock(),
            effective_attn_backend=mock.Mock(return_value="sdpa"),
        )
        snapshot = "/cache/models--owner--model/snapshots/revision"

        with (
            mock.patch.object(LOCAL_PIPELINE, "_import_torch", return_value=fake_torch),
            mock.patch.object(
                LOCAL_PIPELINE,
                "_import_sensenova_u1",
                return_value=(fake_core, load_model_and_tokenizer, mock.Mock()),
            ),
            mock.patch.object(
                LOCAL_PIPELINE,
                "resolve_hf_model_snapshot_interruptibly",
                return_value=snapshot,
            ) as resolve_snapshot,
            mock.patch.object(LOCAL_PIPELINE, "_vram_snapshot"),
        ):
            model = LOCAL_PIPELINE.SenseNovaU1LocalModel(
                model_path="owner/model",
                device="cpu",
                dtype="float32",
            )

        resolve_snapshot.assert_called_once_with("owner/model")
        self.assertEqual(load_model_and_tokenizer.call_args.args[0], snapshot)
        self.assertEqual(model.model_path, "owner/model")


if __name__ == "__main__":
    unittest.main()
