import ast
import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from test_comfyui_model_cache import NODES
from test_comfyui_source_path import LOCAL_PIPELINE

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "apps" / "comfyui" / "nodes.py"
WORKFLOWS_PATH = REPO_ROOT / "apps" / "comfyui" / "example_workflows"


def _class(name: str) -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(node_class: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in node_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


class ComfyUIEditOutputSizeTest(unittest.TestCase):
    def test_size_node_exposes_the_five_semantic_presets(self) -> None:
        self.assertEqual(
            NODES.EDIT_OUTPUT_SIZE_OPTIONS,
            (
                "Auto · 4MP (Recommended)",
                "Auto · 2MP",
                "Auto · 1MP",
                "Match First Input",
                "Custom",
            ),
        )

    def test_size_node_builds_normalized_configs(self) -> None:
        auto = NODES.SenseNovaU1EditOutputSize.execute(
            preset="Auto · 4MP (Recommended)",
            width=2048,
            height=2048,
        ).values[0]
        match_input = NODES.SenseNovaU1EditOutputSize.execute(
            preset="Match First Input",
            width=2048,
            height=2048,
        ).values[0]
        custom = NODES.SenseNovaU1EditOutputSize.execute(
            preset="Custom",
            width=2560,
            height=1600,
        ).values[0]

        self.assertEqual(auto, {"mode": "auto", "target_pixels": 2048 * 2048})
        self.assertEqual(match_input, {"mode": "match_input"})
        self.assertEqual(custom, {"mode": "custom", "width": 2560, "height": 1600})

    def test_connected_custom_size_config_is_normalized(self) -> None:
        resolved = NODES._resolve_edit_output_size(
            {"mode": "custom", "width": 2560, "height": 1600},
        )

        self.assertEqual(resolved, (2560, 1600, None))

    def test_disconnected_size_config_defaults_to_auto_4mp(self) -> None:
        self.assertEqual(
            NODES._resolve_edit_output_size(None),
            (None, None, 2048 * 2048),
        )

    def test_edit_node_forwards_connected_match_input_policy(self) -> None:
        model = mock.Mock()
        model.edit_image.return_value = SimpleNamespace(images="output")

        with (
            mock.patch.object(NODES, "_ensure_local_model_loaded", side_effect=lambda value: value),
            mock.patch.object(NODES, "output_to_tuple", return_value=("output", "", "", "{}")),
            mock.patch.object(NODES, "comfy_image_info", return_value="image-info"),
        ):
            NODES.SenseNovaU1LocalImageEdit.execute(
                u1_model=model,
                image=object(),
                reference_images={},
                prompt="Edit this image.",
                cfg_scale=4.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval_start=0.0,
                cfg_interval_end=1.0,
                num_steps=8,
                batch_size=1,
                seed=42,
                output_size={"mode": "match_input"},
            )

        self.assertIsNone(model.edit_image.call_args.kwargs["width"])
        self.assertIsNone(model.edit_image.call_args.kwargs["height"])
        self.assertIsNone(model.edit_image.call_args.kwargs["target_pixels"])

    def test_edit_schema_exposes_only_the_typed_size_input(self) -> None:
        define_schema = _method(_class("SenseNovaU1LocalImageEdit"), "define_schema")
        schema = next(
            node
            for node in ast.walk(define_schema)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "io.Schema"
        )
        inputs = next(keyword.value for keyword in schema.keywords if keyword.arg == "inputs")
        self.assertIsInstance(inputs, ast.List)
        calls = [node for node in inputs.elts if isinstance(node, ast.Call)]
        by_id = {ast.literal_eval(call.args[0]): call for call in calls if call.args}

        self.assertEqual(ast.unparse(by_id["output_size"].func), "EditOutputSizeIO.Input")
        for input_id in ("auto_size", "width", "height", "target_megapixels"):
            self.assertNotIn(input_id, by_id)

    def test_extension_registers_output_size_node(self) -> None:
        nodes = asyncio.run(NODES.SenseNovaExtension().get_node_list())
        self.assertIn(NODES.SenseNovaU1EditOutputSize, nodes)

    def test_bundled_edit_workflows_drop_legacy_size_widget_values(self) -> None:
        found = 0
        for workflow_path in WORKFLOWS_PATH.glob("*.json"):
            workflow = json.loads(workflow_path.read_text())
            for node in workflow.get("nodes", []):
                if node.get("type") != "SenseNovaU1LocalImageEdit":
                    continue
                found += 1
                values = node["widgets_values"]
                self.assertEqual(len(values), 12, workflow_path)
                self.assertEqual(values[1:4], [4, 1, "none"], workflow_path)
        self.assertGreater(found, 0)

    def test_match_input_uses_original_first_image_pixel_count(self) -> None:
        image = Image.new("RGB", (2272, 3648))
        smart_resize = mock.Mock(return_value=(3648, 2272))

        with mock.patch.object(
            LOCAL_PIPELINE,
            "_import_sensenova_u1",
            return_value=(object(), object(), smart_resize),
        ):
            size = LOCAL_PIPELINE._resolve_edit_size(
                image,
                width=None,
                height=None,
                target_pixels=None,
            )

        self.assertEqual(size, (2272, 3648))
        self.assertEqual(smart_resize.call_args.kwargs["min_pixels"], 2272 * 3648)
        self.assertEqual(smart_resize.call_args.kwargs["max_pixels"], 2272 * 3648)


if __name__ == "__main__":
    unittest.main()
