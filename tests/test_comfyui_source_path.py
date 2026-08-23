import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_APP_DIR = REPO_ROOT / "apps" / "comfyui"
LOCAL_PIPELINE_PATH = COMFYUI_APP_DIR / "local_pipeline.py"


def _load_local_pipeline_module():
    module_name = "_sensenova_comfy_local_pipeline_source_test"
    spec = importlib.util.spec_from_file_location(module_name, LOCAL_PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(sys.modules, {module_name: module}),
        mock.patch.object(sys, "path", [str(COMFYUI_APP_DIR), *sys.path]),
    ):
        spec.loader.exec_module(module)
    return module


LOCAL_PIPELINE = _load_local_pipeline_module()


class ComfyUISourcePathTest(unittest.TestCase):
    def test_default_source_path_does_not_infer_from_module_location(self) -> None:
        with TemporaryDirectory() as directory:
            repo = Path(directory) / "SenseNova-U1"
            module_path = repo / "apps" / "comfyui" / "local_pipeline.py"
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            (repo / "src").mkdir()

            with (
                mock.patch.object(LOCAL_PIPELINE, "__file__", str(module_path)),
                mock.patch.dict("os.environ", {}, clear=True),
            ):
                source_path = LOCAL_PIPELINE.default_source_path()

        self.assertEqual(source_path, "")


if __name__ == "__main__":
    unittest.main()
