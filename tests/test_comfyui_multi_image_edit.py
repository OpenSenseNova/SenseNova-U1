import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from test_comfyui_model_cache import NODES
from test_comfyui_source_path import LOCAL_PIPELINE

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "apps" / "comfyui" / "nodes.py"


def _image_edit_class() -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SenseNovaU1LocalImageEdit"
    )


def _method(node_class: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in node_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


class ComfyUIMultiImageEditTest(unittest.TestCase):
    def test_schema_adds_optional_autogrow_references_after_the_legacy_image_socket(self) -> None:
        define_schema = _method(_image_edit_class(), "define_schema")
        schema = next(
            node
            for node in ast.walk(define_schema)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "io.Schema"
        )
        inputs = next(keyword.value for keyword in schema.keywords if keyword.arg == "inputs")
        self.assertIsInstance(inputs, ast.List)
        top_level_calls = [node for node in inputs.elts if isinstance(node, ast.Call)]
        legacy_image = next(node for node in top_level_calls if ast.unparse(node.func) == "io.Image.Input")
        self.assertEqual(ast.literal_eval(legacy_image.args[0]), "image")

        autogrow = next(
            node
            for node in top_level_calls
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "io.Autogrow.Input"
        )

        self.assertEqual(ast.literal_eval(autogrow.args[0]), "reference_images")
        template = next(keyword.value for keyword in autogrow.keywords if keyword.arg == "template")
        self.assertIsInstance(template, ast.Call)
        self.assertEqual(ast.unparse(template.func), "io.Autogrow.TemplateNames")

        image_input = template.args[0]
        self.assertIsInstance(image_input, ast.Call)
        self.assertEqual(ast.unparse(image_input.func), "io.Image.Input")
        self.assertEqual(ast.literal_eval(image_input.args[0]), "image")

        names = next(keyword.value for keyword in template.keywords if keyword.arg == "names")
        minimum = next(keyword.value for keyword in template.keywords if keyword.arg == "min")
        self.assertEqual(ast.literal_eval(names), [f"image{index}" for index in range(2, 11)])
        self.assertEqual(ast.literal_eval(minimum), 0)

    def test_execute_forwards_autogrow_images_in_socket_order(self) -> None:
        model = mock.Mock()
        model.edit_image.return_value = SimpleNamespace(images="output")
        first_image = object()
        second_image = object()

        with (
            mock.patch.object(NODES, "_ensure_local_model_loaded", side_effect=lambda value: value),
            mock.patch.object(NODES, "output_to_tuple", return_value=("output", "", "", "{}")),
            mock.patch.object(NODES, "comfy_image_info", return_value="image-info"),
        ):
            NODES.SenseNovaU1LocalImageEdit.execute(
                u1_model=model,
                image=first_image,
                reference_images={"image2": second_image},
                prompt="Use the second image as the style reference.",
                auto_size=True,
                width=2048,
                height=2048,
                target_megapixels=4.194304,
                cfg_scale=4.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval_start=0.0,
                cfg_interval_end=1.0,
                num_steps=8,
                batch_size=1,
                seed=42,
            )

        self.assertEqual(
            model.edit_image.call_args.kwargs["input_images"],
            [first_image, second_image],
        )

    def test_pipeline_flattens_image_batches_in_socket_order(self) -> None:
        first_batch = np.zeros((1, 2, 3, 3), dtype=np.float32)
        first_batch[0, ..., 0] = 1.0
        second_batch = np.zeros((2, 2, 3, 3), dtype=np.float32)
        second_batch[0, ..., 1] = 1.0
        second_batch[1, ..., 2] = 1.0

        offloaded = mock.Mock()
        offloaded.it2i_generate.return_value = "generated-tensor"
        local_model = SimpleNamespace(
            tokenizer="tokenizer",
            model=object(),
            info={"model_path": "model"},
            _offload_ctx=lambda: contextlib.nullcontext(offloaded),
        )
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)

        with (
            mock.patch.object(LOCAL_PIPELINE, "_import_torch", return_value=fake_torch),
            mock.patch.object(
                LOCAL_PIPELINE,
                "_progress_hook",
                side_effect=lambda *_args: contextlib.nullcontext(),
            ),
            mock.patch.object(
                LOCAL_PIPELINE,
                "_resize_input_to_budget",
                side_effect=lambda image, _target_pixels: image,
            ),
            mock.patch.object(LOCAL_PIPELINE, "_resolve_edit_size", return_value=(32, 32)),
            mock.patch.object(LOCAL_PIPELINE, "_batch_tensor_to_comfy_image", return_value="output"),
        ):
            result = LOCAL_PIPELINE.SenseNovaU1LocalModel.edit_image(
                local_model,
                prompt="Combine the references.",
                input_images=[first_batch, second_batch],
                width=None,
                height=None,
                target_pixels=32 * 32,
                cfg_scale=4.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval=(0.0, 1.0),
                num_steps=8,
                batch_size=1,
                seed=42,
                think_mode=False,
            )

        forwarded = offloaded.it2i_generate.call_args.args[2]
        self.assertEqual(len(forwarded), 3)
        self.assertEqual([image.getpixel((0, 0)) for image in forwarded], [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        self.assertEqual(result.metadata["input_image_count"], 3)

    def test_pipeline_separates_multi_image_input_budget_from_output_target(self) -> None:
        input_batches = [object(), object(), object()]
        pil_images = [object(), object(), object()]
        offloaded = mock.Mock()
        offloaded.it2i_generate.return_value = "generated-tensor"
        local_model = SimpleNamespace(
            tokenizer="tokenizer",
            model=object(),
            info={"model_path": "model"},
            _offload_ctx=lambda: contextlib.nullcontext(offloaded),
        )
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        output_target_pixels = 4_000_000
        terminal_auto_input_pixels = (2 * 2048 * 2048) // len(pil_images)

        with (
            mock.patch.object(LOCAL_PIPELINE, "_import_torch", return_value=fake_torch),
            mock.patch.object(
                LOCAL_PIPELINE,
                "comfy_batch_to_pil_images",
                side_effect=lambda batch: [pil_images[input_batches.index(batch)]],
            ),
            mock.patch.object(
                LOCAL_PIPELINE,
                "_progress_hook",
                side_effect=lambda *_args: contextlib.nullcontext(),
            ),
            mock.patch.object(
                LOCAL_PIPELINE,
                "_resize_input_to_budget",
                side_effect=lambda image, _input_pixels: image,
            ) as resize_input,
            mock.patch.object(LOCAL_PIPELINE, "_resolve_edit_size", return_value=(2560, 1600)) as resolve_size,
            mock.patch.object(LOCAL_PIPELINE, "_batch_tensor_to_comfy_image", return_value="output"),
        ):
            LOCAL_PIPELINE.SenseNovaU1LocalModel.edit_image(
                local_model,
                prompt="Combine the references.",
                input_images=input_batches,
                width=None,
                height=None,
                target_pixels=output_target_pixels,
                cfg_scale=1.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval=(0.0, 1.0),
                num_steps=8,
                batch_size=1,
                seed=42,
                think_mode=False,
            )

        self.assertEqual(
            [call.args[1] for call in resize_input.call_args_list],
            [terminal_auto_input_pixels] * len(pil_images),
        )
        self.assertEqual(resolve_size.call_args.kwargs["target_pixels"], output_target_pixels)


if __name__ == "__main__":
    unittest.main()
