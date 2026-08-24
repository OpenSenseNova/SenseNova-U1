import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_comfyui_model_cache import NODES

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "apps" / "comfyui" / "nodes.py"


def _prompt_builder_class() -> ast.ClassDef:
    tree = ast.parse(NODES_PATH.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SenseNovaPromptBuilder")


def _method(node_class: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in node_class.body if isinstance(node, ast.FunctionDef) and node.name == name)


class ComfyUIPromptBuilderImagesTest(unittest.TestCase):
    def test_schema_exposes_ten_optional_autogrow_image_inputs(self) -> None:
        define_schema = _method(_prompt_builder_class(), "define_schema")
        schema = next(
            node
            for node in ast.walk(define_schema)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "io.Schema"
        )
        inputs = next(keyword.value for keyword in schema.keywords if keyword.arg == "inputs")
        self.assertIsInstance(inputs, ast.List)
        autogrow = next(
            node for node in inputs.elts if isinstance(node, ast.Call) and ast.unparse(node.func) == "io.Autogrow.Input"
        )

        self.assertEqual(ast.literal_eval(autogrow.args[0]), "images")
        template = next(keyword.value for keyword in autogrow.keywords if keyword.arg == "template")
        self.assertEqual(ast.unparse(template.func), "io.Autogrow.TemplateNames")
        self.assertEqual(ast.unparse(template.args[0].func), "io.Image.Input")
        self.assertEqual(ast.literal_eval(template.args[0].args[0]), "image")
        names = next(keyword.value for keyword in template.keywords if keyword.arg == "names")
        minimum = next(keyword.value for keyword in template.keywords if keyword.arg == "min")
        self.assertEqual(ast.literal_eval(names), [f"image{index}" for index in range(1, 11)])
        self.assertEqual(ast.literal_eval(minimum), 0)

    def test_execute_flattens_image_batches_and_preserves_socket_order(self) -> None:
        first_batch = object()
        second_batch = object()
        first_image = object()
        second_image = object()
        third_image = object()
        client = mock.MagicMock()
        client.chat.return_value = SimpleNamespace(text="expanded", usage={}, raw={})
        client_type = mock.MagicMock()
        client_type.from_env.return_value = client

        with (
            mock.patch.object(NODES, "SenseNovaClient", client_type),
            mock.patch.object(
                NODES,
                "comfy_batch_to_pil_images",
                side_effect=[[first_image], [second_image, third_image]],
            ),
            mock.patch.object(
                NODES,
                "pil_to_png_data_url",
                side_effect=lambda image: f"data:{id(image)}",
            ),
            mock.patch.object(
                NODES,
                "_list_multimodal_chat_model_options",
                return_value=("sensenova-6.7-flash-lite",),
            ),
        ):
            NODES.SenseNovaPromptBuilder.execute(
                prompt="Use all references.",
                system_prompt="Expand the prompt.",
                model="sensenova-6.7-flash-lite",
                temperature=0.3,
                top_p=1.0,
                max_tokens=100,
                timeout=30,
                images={"image1": first_batch, "image2": second_batch},
            )

        self.assertEqual(
            client.chat.call_args.kwargs["image_urls"],
            [f"data:{id(first_image)}", f"data:{id(second_image)}", f"data:{id(third_image)}"],
        )

    def test_execute_rejects_more_than_ten_images_after_batch_expansion(self) -> None:
        with (
            mock.patch.object(NODES, "comfy_batch_to_pil_images", return_value=[object()] * 11),
            self.assertRaisesRegex(RuntimeError, "at most 10 images"),
        ):
            NODES.SenseNovaPromptBuilder.execute(
                prompt="Use all references.",
                system_prompt="Expand the prompt.",
                model="sensenova-6.7-flash-lite",
                temperature=0.3,
                top_p=1.0,
                max_tokens=100,
                timeout=30,
                images={"image1": object()},
            )

    def test_execute_rejects_a_text_only_model_when_images_are_connected(self) -> None:
        with (
            mock.patch.object(NODES, "comfy_batch_to_pil_images", return_value=[object()]),
            mock.patch.object(NODES, "pil_to_png_data_url", return_value="data:image/png;base64,aW1hZ2U="),
            mock.patch.object(
                NODES,
                "_list_multimodal_chat_model_options",
                return_value=("sensenova-6.7-flash-lite",),
            ),
            self.assertRaisesRegex(RuntimeError, "does not support image input"),
        ):
            NODES.SenseNovaPromptBuilder.execute(
                prompt="Use the reference.",
                system_prompt="Expand the prompt.",
                model="deepseek-v4-flash",
                temperature=0.3,
                top_p=1.0,
                max_tokens=100,
                timeout=30,
                images={"image1": object()},
            )


if __name__ == "__main__":
    unittest.main()
