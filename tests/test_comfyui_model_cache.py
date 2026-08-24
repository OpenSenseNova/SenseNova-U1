import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_APP_DIR = REPO_ROOT / "apps" / "comfyui"
NODES_PATH = COMFYUI_APP_DIR / "nodes.py"


class _FakeCustomType:
    @staticmethod
    def Input(*args, **kwargs):
        return None

    @staticmethod
    def Output(*args, **kwargs):
        return None


class _FakeNodeOutput:
    def __init__(self, *values, **kwargs):
        self.values = values


class _RecordingWidgetType:
    class Input:
        def __init__(self, input_id, *args, **kwargs):
            self.id = input_id
            self.options = kwargs.get("options")
            self.default = kwargs.get("default")

    class Output:
        def __init__(self, *args, **kwargs):
            pass


class _RecordingStringInput(_RecordingWidgetType.Input):
    pass


class _RecordingStringType:
    Input = _RecordingStringInput
    Output = _RecordingWidgetType.Output


class _RecordingComboInput(_RecordingWidgetType.Input):
    pass


class _RecordingComboType:
    Input = _RecordingComboInput
    Output = _RecordingWidgetType.Output


class _RecordingSchema:
    def __init__(self, **kwargs):
        self.inputs = kwargs["inputs"]


def _load_nodes_module():
    fake_io = types.SimpleNamespace(
        ComfyNode=object,
        NodeOutput=_FakeNodeOutput,
        Custom=lambda _name: _FakeCustomType(),
    )
    fake_latest = types.ModuleType("comfy_api.latest")
    fake_latest.ComfyExtension = object
    fake_latest.io = fake_io
    fake_comfy_api = types.ModuleType("comfy_api")
    fake_comfy_api.latest = fake_latest

    fake_api_client = types.ModuleType("api_client")
    fake_api_client.CHAT_MODELS = ("chat",)
    fake_api_client.IMAGE_MODELS = ("image",)
    fake_api_client.IMAGE_SIZE_OPTIONS = ("1024x1024",)
    fake_api_client.VISION_MODELS = ("vision",)
    fake_api_client.SenseNovaClient = object

    fake_image_utils = types.ModuleType("image_utils")
    for name in (
        "comfy_batch_to_pil_images",
        "comfy_image_info",
        "comfy_image_to_png_data_url",
        "image_bytes_to_comfy_image",
    ):
        setattr(fake_image_utils, name, lambda *args, **kwargs: None)

    fake_local_pipeline = types.ModuleType("local_pipeline")
    local_pipeline_values = {
        "ATTN_BACKEND_OPTIONS": ("auto",),
        "CFG_NORM_OPTIONS": ("none", "global"),
        "DEFAULT_FAST_ACTIVATION_RESERVE_GIB": 4.0,
        "DEFAULT_FAST_VRAM_FRACTION": 0.9,
        "DEFAULT_FAST_VRAM_HEADROOM_GIB": 2.0,
        "DEFAULT_INTERLEAVE_SYSTEM_MESSAGE": "",
        "DEFAULT_SEED": 42,
        "DEFAULT_VRAM_MODE": "full",
        "DEVICE_MAP_OPTIONS": ("none",),
        "DTYPE_OPTIONS": ("bfloat16",),
        "INTERLEAVE_RESOLUTION_OPTIONS": ("1024x1024|1:1", "2048x1152|16:9"),
        "INTERLEAVE_RESULT_TYPE": "INTERLEAVE_RESULT",
        "LOCAL_MODEL_TYPE": "LOCAL_MODEL",
        "T2I_RESOLUTION_OPTIONS": ("2048x2048|1:1",),
        "VRAM_MODE_OPTIONS": ("full",),
        "SenseNovaU1LocalModel": object,
    }
    for name, value in local_pipeline_values.items():
        setattr(fake_local_pipeline, name, value)
    for name in (
        "default_device",
        "default_source_path",
        "interleave_output_to_tuple",
        "interleave_result_to_markdown",
        "output_to_tuple",
        "parse_resolution_option",
    ):
        setattr(fake_local_pipeline, name, lambda *args, **kwargs: None)

    fake_prompt_utils = types.ModuleType("prompt_utils")
    fake_prompt_utils.load_prompt_template = lambda *args, **kwargs: ""

    module_name = "_sensenova_comfy_nodes_cache_test"
    spec = importlib.util.spec_from_file_location(module_name, NODES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(
            sys.modules,
            {
                "comfy_api": fake_comfy_api,
                "comfy_api.latest": fake_latest,
                "api_client": fake_api_client,
                "image_utils": fake_image_utils,
                "local_pipeline": fake_local_pipeline,
                module_name: module,
                "prompt_utils": fake_prompt_utils,
            },
        ),
        mock.patch.object(sys, "path", [str(COMFYUI_APP_DIR), *sys.path]),
    ):
        spec.loader.exec_module(module)
    return module


NODES = _load_nodes_module()


def _cache_key(model_path: str) -> tuple:
    return (
        model_path,
        "",
        "",
        "cuda",
        "bfloat16",
        "auto",
        "none",
        "",
        "full",
        0.9,
        2.0,
        4.0,
        0.0,
        "",
        ("", 0, 0),
        1.0,
    )


class _FakeLocalModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = object()
        self.tokenizer = object()
        self.info = {"model_path": kwargs["model_path"]}
        self.__class__.instances.append(self)


class ComfyUILocalModelCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        NODES._LOCAL_MODEL_CACHE.clear()
        _FakeLocalModel.instances.clear()

    def tearDown(self) -> None:
        NODES._LOCAL_MODEL_CACHE.clear()

    def test_stale_comfyui_output_reloads_its_original_model(self) -> None:
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with (
            mock.patch.object(NODES, "SenseNovaU1LocalModel", _FakeLocalModel),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            old_a = NODES._get_or_load_local_model(_cache_key("model-a"))
            live_b = NODES._get_or_load_local_model(_cache_key("model-b"))

            self.assertFalse(hasattr(old_a, "model"))
            self.assertFalse(hasattr(old_a, "tokenizer"))
            self.assertIs(NODES._LOCAL_MODEL_CACHE[_cache_key("model-b")], live_b)

            restored_a = NODES._ensure_local_model_loaded(old_a)

            self.assertIsNot(restored_a, old_a)
            self.assertEqual(restored_a.info["model_path"], "model-a")
            self.assertTrue(hasattr(restored_a, "model"))
            self.assertTrue(hasattr(restored_a, "tokenizer"))
            self.assertFalse(hasattr(live_b, "model"))
            self.assertIs(NODES._LOCAL_MODEL_CACHE[_cache_key("model-a")], restored_a)

    def test_stale_alias_resolves_to_already_restored_model(self) -> None:
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with (
            mock.patch.object(NODES, "SenseNovaU1LocalModel", _FakeLocalModel),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            stale_a = NODES._get_or_load_local_model(_cache_key("model-a"))
            NODES._get_or_load_local_model(_cache_key("model-b"))
            restored_a = NODES._ensure_local_model_loaded(stale_a)

            resolved_again = NODES._ensure_local_model_loaded(stale_a)

            self.assertIs(resolved_again, restored_a)
            self.assertEqual(len(_FakeLocalModel.instances), 3)

    def test_loader_schema_lists_complete_models_from_registered_sensenova_folders(self) -> None:
        with TemporaryDirectory() as directory:
            model_root = Path(directory) / "sensenova"
            complete_model = model_root / "release" / "SenseNova-U1.5-8B-MoT"
            complete_model.mkdir(parents=True)
            (complete_model / "config.json").write_text("{}")
            (complete_model / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}'
            )
            (complete_model / "model-00001-of-00001.safetensors").touch()
            incomplete_model = model_root / "incomplete"
            incomplete_model.mkdir()
            (incomplete_model / "config.json").write_text("{}")
            (incomplete_model / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"missing.safetensors"}}'
            )
            (model_root / "community.safetensors").touch()
            (model_root / "quantized.gguf").touch()

            fake_folder_paths = types.SimpleNamespace(
                get_filename_list=lambda _name: [],
                get_folder_paths=lambda name: [str(model_root)] if name == "sensenova" else [],
            )
            recording_io = types.SimpleNamespace(
                Schema=_RecordingSchema,
                String=_RecordingWidgetType,
                Combo=_RecordingWidgetType,
                Float=_RecordingWidgetType,
                Int=_RecordingWidgetType,
            )
            with (
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
                mock.patch.object(NODES, "io", recording_io),
            ):
                schema = NODES.SenseNovaU1LocalLoader.define_schema()

        local_model_input = next(item for item in schema.inputs if item.id == "local_model")
        self.assertEqual(
            local_model_input.options,
            ["", "community.safetensors", "quantized.gguf", "release/SenseNova-U1.5-8B-MoT"],
        )

    def test_new_loader_separates_weights_from_resources(self) -> None:
        recording_io = types.SimpleNamespace(
            Schema=_RecordingSchema,
            String=_RecordingWidgetType,
            Combo=_RecordingWidgetType,
            Float=_RecordingWidgetType,
            Int=_RecordingWidgetType,
        )
        with (
            mock.patch.object(
                NODES,
                "_list_model_weight_options",
                return_value=["HF | sensenova/SenseNova-U1.5-8B-MoT", "Local | community.sft"],
            ),
            mock.patch.object(
                NODES,
                "_list_model_resource_options",
                return_value=["Auto", "HF | sensenova/SenseNova-U1.5-8B-MoT"],
            ),
            mock.patch.object(NODES, "io", recording_io),
        ):
            schema = NODES.SenseNovaU1ModelLoader.define_schema()

        inputs = {item.id: item for item in schema.inputs}
        self.assertEqual(
            inputs["model_weights"].options,
            ["HF | sensenova/SenseNova-U1.5-8B-MoT", "Local | community.sft"],
        )
        self.assertEqual(
            inputs["model_resources"].options,
            ["Auto", "HF | sensenova/SenseNova-U1.5-8B-MoT"],
        )
        self.assertIn("lora_name", inputs)
        self.assertIn("lora_strength", inputs)

    def test_new_loader_accepts_lora_name_as_a_string_input(self) -> None:
        recording_io = types.SimpleNamespace(
            Schema=_RecordingSchema,
            String=_RecordingStringType,
            Combo=_RecordingComboType,
            Float=_RecordingWidgetType,
            Int=_RecordingWidgetType,
        )
        with (
            mock.patch.object(
                NODES,
                "_list_model_weight_options",
                return_value=["HF | sensenova/SenseNova-U1.5-8B-MoT"],
            ),
            mock.patch.object(
                NODES,
                "_list_model_resource_options",
                return_value=["Auto"],
            ),
            mock.patch.object(NODES, "io", recording_io),
        ):
            schema = NODES.SenseNovaU1ModelLoader.define_schema()

        lora_input = next(item for item in schema.inputs if item.id == "lora_name")
        self.assertIsInstance(lora_input, _RecordingStringInput)

    def test_lora_selector_lists_files_outputs_selection_and_is_registered(self) -> None:
        options = ["", "adapter.safetensors"]
        recording_io = types.SimpleNamespace(
            Schema=_RecordingSchema,
            String=_RecordingStringType,
            Combo=_RecordingComboType,
        )
        with (
            mock.patch.object(NODES, "_list_lora_options", return_value=options),
            mock.patch.object(NODES, "io", recording_io),
        ):
            schema = NODES.SenseNovaU1LoraSelector.define_schema()

        selector_input = next(item for item in schema.inputs if item.id == "lora_name")
        self.assertIsInstance(selector_input, _RecordingComboInput)
        self.assertEqual(selector_input.options, options)

        output = NODES.SenseNovaU1LoraSelector.execute("adapter.safetensors")
        self.assertEqual(output.values, ("adapter.safetensors",))
        self.assertIn(
            NODES.SenseNovaU1LoraSelector,
            asyncio.run(NODES.SenseNovaExtension().get_node_list()),
        )

    def test_new_loader_passes_explicit_resources_to_the_core_model(self) -> None:
        inputs = {
            "model_weights": "HF | community/repacked-u1",
            "model_resources": "HF | sensenova/SenseNova-U1.5-8B-MoT",
            "lora_name": "",
            "lora_strength": 1.0,
            "device": "cpu",
            "dtype": "bfloat16",
            "attn_backend": "auto",
            "device_map": "none",
            "max_memory": "",
            "vram_mode": "full",
        }
        with mock.patch.object(NODES, "SenseNovaU1LocalModel", _FakeLocalModel):
            output = NODES.SenseNovaU1ModelLoader.execute(**inputs)

        loaded_model = output.values[0]
        self.assertEqual(loaded_model.kwargs["model_path"], "community/repacked-u1")
        self.assertEqual(
            loaded_model.kwargs["model_resources"],
            "sensenova/SenseNova-U1.5-8B-MoT",
        )

    def test_new_loader_lists_cached_hf_and_registered_local_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            model_root = Path(directory) / "sensenova"
            model_root.mkdir()
            checkpoint = model_root / "community.sft"
            checkpoint.touch()
            resources = model_root / "resources"
            resources.mkdir()
            (resources / "config.json").write_text("{}")
            fake_folder_paths = types.SimpleNamespace(
                get_folder_paths=lambda name: [str(model_root)] if name == "sensenova" else [],
            )
            with (
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
                mock.patch.object(NODES, "_cached_sensenova_repo_ids", return_value={"community/cached-sensenova"}),
            ):
                weight_options = NODES._list_model_weight_options()
                resource_options = NODES._list_model_resource_options()
                resolved_weight = NODES._resolve_model_weight_choice("Local | community.sft")
                resolved_resources = NODES._resolve_model_resource_choice("Local | resources")

        self.assertIn("HF | community/cached-sensenova", weight_options)
        self.assertIn("Local | community.sft", weight_options)
        self.assertIn("Local | resources", resource_options)
        self.assertEqual(resolved_weight, str(checkpoint.resolve()))
        self.assertEqual(resolved_resources, str(resources.resolve()))

    def test_loader_resolves_a_registered_single_file_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            model_root = Path(directory) / "sensenova"
            model_root.mkdir()
            checkpoint = model_root / "community.safetensors"
            checkpoint.touch()
            fake_folder_paths = types.SimpleNamespace(
                get_folder_paths=lambda name: [str(model_root)] if name == "sensenova" else [],
            )

            with mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
                resolved = NODES._resolve_sensenova_model_choice("community.safetensors")

        self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_loader_resolves_a_registered_sensenova_model_before_loading(self) -> None:
        with TemporaryDirectory() as directory:
            model_root = Path(directory) / "sensenova"
            model_directory = model_root / "SenseNova-U1.5-8B-MoT"
            model_directory.mkdir(parents=True)
            (model_directory / "config.json").write_text("{}")
            (model_directory / "model.safetensors").touch()
            fake_folder_paths = types.SimpleNamespace(
                get_folder_paths=lambda name: [str(model_root)] if name == "sensenova" else [],
            )

            with (
                mock.patch.object(NODES, "SenseNovaU1LocalModel", _FakeLocalModel),
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
            ):
                first_fingerprint = NODES.SenseNovaU1LocalLoader.fingerprint_inputs(
                    model_path="remote/model-a",
                    sensenova_u1_src="",
                    device="cpu",
                    dtype="bfloat16",
                    attn_backend="auto",
                    device_map="none",
                    max_memory="",
                    vram_mode="full",
                    gguf_checkpoint="",
                    local_model="SenseNova-U1.5-8B-MoT",
                )
                second_fingerprint = NODES.SenseNovaU1LocalLoader.fingerprint_inputs(
                    model_path="remote/model-b",
                    sensenova_u1_src="",
                    device="cpu",
                    dtype="bfloat16",
                    attn_backend="auto",
                    device_map="none",
                    max_memory="",
                    vram_mode="full",
                    gguf_checkpoint="",
                    local_model="SenseNova-U1.5-8B-MoT",
                )
                output = NODES.SenseNovaU1LocalLoader.execute(
                    model_path="sensenova/SenseNova-U1-8B-MoT",
                    sensenova_u1_src="",
                    device="cpu",
                    dtype="bfloat16",
                    attn_backend="auto",
                    device_map="none",
                    max_memory="",
                    vram_mode="full",
                    gguf_checkpoint="",
                    local_model="SenseNova-U1.5-8B-MoT",
                )

        loaded_model = output.values[0]
        self.assertEqual(first_fingerprint, second_fingerprint)
        self.assertEqual(loaded_model.kwargs["model_path"], str(model_directory.resolve()))

    def test_loader_passes_lora_and_fingerprints_the_file_contents(self) -> None:
        with TemporaryDirectory() as directory:
            lora_path = Path(directory) / "adapter.safetensors"
            lora_path.write_bytes(b"v1")
            fake_folder_paths = types.SimpleNamespace(
                get_full_path=lambda name, value: (
                    str(lora_path) if (name, value) == ("loras", lora_path.name) else None
                ),
            )
            inputs = {
                "model_path": "sensenova/SenseNova-U1.5-8B-MoT",
                "sensenova_u1_src": "",
                "device": "cpu",
                "dtype": "bfloat16",
                "attn_backend": "auto",
                "device_map": "none",
                "max_memory": "",
                "vram_mode": "full",
                "gguf_checkpoint": "",
                "lora_name": lora_path.name,
                "lora_strength": 0.75,
            }
            with (
                mock.patch.object(NODES, "SenseNovaU1LocalModel", _FakeLocalModel),
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
            ):
                first_fingerprint = NODES.SenseNovaU1LocalLoader.fingerprint_inputs(**inputs)
                output = NODES.SenseNovaU1LocalLoader.execute(**inputs)
                lora_path.write_bytes(b"version-two")
                second_fingerprint = NODES.SenseNovaU1LocalLoader.fingerprint_inputs(**inputs)

        loaded_model = output.values[0]
        self.assertEqual(loaded_model.kwargs["lora_path"], str(lora_path))
        self.assertEqual(loaded_model.kwargs["lora_strength"], 0.75)
        self.assertNotEqual(first_fingerprint, second_fingerprint)


if __name__ == "__main__":
    unittest.main()
