import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from sensenova_u1.utils.checkpoint_loading import load_model_and_tokenizer


class _FakeModel:
    def eval(self):
        return self


class CheckpointLoadingTest(unittest.TestCase):
    def test_hf_id_is_preserved_when_cached_snapshot_has_no_weights(self) -> None:
        model_id = "example/incomplete-cache"
        calls: list[tuple[str, str]] = []
        config = object()
        tokenizer = object()

        def load_config(path: str):
            calls.append(("config", path))
            return config

        def load_tokenizer(path: str):
            calls.append(("tokenizer", path))
            return tokenizer

        def load_model(path: str, **_kwargs):
            calls.append(("model", path))
            return _FakeModel()

        with TemporaryDirectory() as directory:
            incomplete_snapshot = Path(directory)
            (incomplete_snapshot / "config.json").write_text("{}")
            with (
                mock.patch("huggingface_hub.snapshot_download", return_value=str(incomplete_snapshot)),
                mock.patch("transformers.AutoConfig.from_pretrained", side_effect=load_config),
                mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=load_tokenizer),
                mock.patch("transformers.AutoModel.from_pretrained", side_effect=load_model),
                mock.patch("sensenova_u1.check_checkpoint_compatibility"),
            ):
                model, loaded_tokenizer = load_model_and_tokenizer(
                    model_id,
                    dtype=torch.bfloat16,
                    for_offload=True,
                )

        self.assertIsInstance(model, _FakeModel)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(calls, [("config", model_id), ("tokenizer", model_id), ("model", model_id)])

    def test_hf_id_is_preserved_when_cached_snapshot_is_missing_a_weight_shard(self) -> None:
        model_id = "example/partial-shards"
        calls: list[str] = []

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0": "model-00001-of-00002.safetensors",
                            "layer.1": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )
            (snapshot / "model-00001-of-00002.safetensors").touch()
            with (
                mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
                mock.patch(
                    "transformers.AutoConfig.from_pretrained", side_effect=lambda path: calls.append(path) or object()
                ),
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
                mock.patch("transformers.AutoModel.from_pretrained", return_value=_FakeModel()),
                mock.patch("sensenova_u1.check_checkpoint_compatibility"),
            ):
                load_model_and_tokenizer(model_id, dtype=torch.bfloat16, for_offload=True)

        self.assertEqual(calls, [model_id])

    def test_complete_cached_snapshot_remains_available_offline(self) -> None:
        model_id = "example/complete-cache"
        calls: list[str] = []

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "model.safetensors").touch()
            with (
                mock.patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
                mock.patch(
                    "transformers.AutoConfig.from_pretrained", side_effect=lambda path: calls.append(path) or object()
                ),
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
                mock.patch("transformers.AutoModel.from_pretrained", return_value=_FakeModel()),
                mock.patch("sensenova_u1.check_checkpoint_compatibility"),
            ):
                load_model_and_tokenizer(model_id, dtype=torch.bfloat16, for_offload=True)

        self.assertEqual(calls, [str(snapshot)])


if __name__ == "__main__":
    unittest.main()
