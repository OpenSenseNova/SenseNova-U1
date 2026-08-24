from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
API_CLIENT_PATH = REPO_ROOT / "apps" / "comfyui" / "api_client.py"


@dataclass(frozen=True)
class _FakeConfig:
    api_key: str
    base_url: str


def _load_api_client_module():
    fake_config = types.ModuleType("config")
    fake_config.SenseNovaConfig = _FakeConfig
    fake_config.load_config = lambda: _FakeConfig(
        api_key="from-env",
        base_url="https://token.sensenova.cn/v1",
    )

    fake_image_utils = types.ModuleType("image_utils")
    fake_image_utils.MAX_IMAGE_BYTES = 50 * 1024 * 1024
    fake_image_utils.is_http_url = lambda value: value.startswith(("http://", "https://"))
    fake_image_utils.is_supported_vision_image_url = lambda _value: True
    fake_image_utils.strip_data_url = lambda value: value

    module_name = "_sensenova_comfy_api_client_models_test"
    spec = importlib.util.spec_from_file_location(module_name, API_CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "config": fake_config,
            "image_utils": fake_image_utils,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


API_CLIENT = _load_api_client_module()


class SenseNovaApiModelsTest(unittest.TestCase):
    def test_extract_model_catalog_filters_by_input_and_output_modalities(self) -> None:
        raw = {
            "object": "list",
            "data": [
                {
                    "id": "vision-chat",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                {"id": "text-chat", "input_modalities": ["text"], "output_modalities": ["text"]},
                {"id": "image-a", "input_modalities": ["text"], "output_modalities": ["image"]},
                {"id": "image-a", "input_modalities": ["text"], "output_modalities": ["image"]},
                {"id": "image-output-only", "output_modalities": ["image"]},
                {"id": "", "output_modalities": ["image"]},
                {"name": "missing-id", "output_modalities": ["image"]},
                "malformed",
            ],
        }

        catalog = API_CLIENT._extract_model_catalog(raw)

        self.assertEqual(catalog.chat_models, ("vision-chat", "text-chat"))
        self.assertEqual(catalog.image_models, ("image-a", "image-output-only"))

    def test_list_models_uses_shared_config_key_and_models_endpoint(self) -> None:
        response = mock.MagicMock()
        response.json.return_value = {
            "data": [
                {"id": "text-chat", "input_modalities": ["text"], "output_modalities": ["text"]},
                {"id": "image-a", "input_modalities": ["text"], "output_modalities": ["image"]},
            ],
        }
        http_client = mock.MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.return_value = response
        client = API_CLIENT.SenseNovaClient(_FakeConfig(api_key="shared-key", base_url="https://token.sensenova.cn/v1"))

        with mock.patch.object(API_CLIENT.httpx, "Client", return_value=http_client) as client_factory:
            catalog = client.list_models(timeout=7)

        self.assertEqual(catalog.chat_models, ("text-chat",))
        self.assertEqual(catalog.image_models, ("image-a",))
        client_factory.assert_called_once_with(timeout=7)
        http_client.get.assert_called_once_with(
            "https://token.sensenova.cn/v1/models",
            headers={"Authorization": "Bearer shared-key"},
        )
        response.raise_for_status.assert_called_once_with()

    def test_chat_accepts_a_model_discovered_at_runtime(self) -> None:
        client = API_CLIENT.SenseNovaClient(_FakeConfig(api_key="shared-key", base_url="https://token.sensenova.cn/v1"))
        raw = {"choices": [{"message": {"content": "expanded prompt"}}]}

        with mock.patch.object(client, "_post_json", return_value=raw) as post_json:
            result = client.chat(
                text="draw a cat",
                system_prompt="expand prompts",
                model="deepseek-v4-flash",
                temperature=0.3,
                top_p=1.0,
                max_tokens=100,
                timeout=30,
            )

        self.assertEqual(result.text, "expanded prompt")
        self.assertEqual(post_json.call_args.args[1]["model"], "deepseek-v4-flash")

    def test_generate_image_accepts_a_model_discovered_at_runtime(self) -> None:
        client = API_CLIENT.SenseNovaClient(_FakeConfig(api_key="shared-key", base_url="https://token.sensenova.cn/v1"))
        raw = {"data": [{"url": "https://example.invalid/image.png"}]}

        with (
            mock.patch.object(client, "_post_json", return_value=raw) as post_json,
            mock.patch.object(client, "download_image", return_value=b"image"),
        ):
            result = client.generate_image(
                prompt="draw a cat",
                model="new-image-model",
                size="2048x2048|1:1",
                timeout=30,
            )

        self.assertEqual(result.image_bytes, b"image")
        self.assertEqual(post_json.call_args.args[0], "/images/generations")
        self.assertEqual(post_json.call_args.args[1]["model"], "new-image-model")

    def test_generate_image_rejects_an_empty_model(self) -> None:
        client = API_CLIENT.SenseNovaClient(_FakeConfig(api_key="shared-key", base_url="https://token.sensenova.cn/v1"))

        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            client.generate_image(
                prompt="draw a cat",
                model="   ",
                size="2048x2048",
                timeout=30,
            )


if __name__ == "__main__":
    unittest.main()
