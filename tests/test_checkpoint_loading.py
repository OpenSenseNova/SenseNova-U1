import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from safetensors.torch import save_file

from sensenova_u1.utils.checkpoint_loading import load_model_and_tokenizer, resolve_model_artifact


class _FakeModel:
    def eval(self):
        return self


class CheckpointLoadingTest(unittest.TestCase):
    def test_single_safetensors_uses_embedded_source_repo_for_resources(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "SenseNova-U1.5-8B-MoT-T8.safetensors"
            save_file(
                {"weight": torch.ones(1)},
                checkpoint,
                metadata={
                    "format": "sensenova-u1.5-mot",
                    "source_repo": "sensenova/SenseNova-U1.5-8B-MoT",
                },
            )

            artifact = resolve_model_artifact(str(checkpoint))

        self.assertEqual(artifact.format, "safetensors")
        self.assertEqual(artifact.weights_path, str(checkpoint))
        self.assertEqual(artifact.resources_path, "sensenova/SenseNova-U1.5-8B-MoT")
        self.assertEqual(artifact.metadata["format"], "sensenova-u1.5-mot")

    def test_single_gguf_is_recognized_without_separate_checkpoint_input(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "SenseNova-U1.5-8B-MoT-Q4_K_M.gguf"
            checkpoint.touch()

            artifact = resolve_model_artifact(str(checkpoint))

        self.assertEqual(artifact.format, "gguf")
        self.assertEqual(artifact.weights_path, str(checkpoint))
        self.assertEqual(artifact.resources_path, "sensenova/SenseNova-U1.5-8B-MoT")

    def test_preview_gguf_infers_the_preview_resources(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "SenseNova-U1.5-8B-MoT-Preview-Q8.gguf"
            checkpoint.touch()

            artifact = resolve_model_artifact(str(checkpoint))

        self.assertEqual(artifact.resources_path, "sensenova/SenseNova-U1.5-8B-MoT-Preview")

    def test_single_safetensors_prefers_a_sibling_resource_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "SenseNova-U1.5-8B-MoT"
            resources.mkdir()
            (resources / "config.json").write_text("{}")
            checkpoint = root / "community.safetensors"
            save_file(
                {"weight": torch.ones(1)},
                checkpoint,
                metadata={"source_repo": "sensenova/SenseNova-U1.5-8B-MoT"},
            )

            artifact = resolve_model_artifact(str(checkpoint))

        self.assertEqual(artifact.resources_path, str(resources))

    def test_community_sft_streams_into_a_meta_initialized_model(self) -> None:
        resource_id = "sensenova/SenseNova-U1.5-8B-MoT"
        config = object()
        tokenizer = object()

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "SenseNova-U1.5-8B-MoT.sft"
            save_file(
                {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                checkpoint,
                metadata={
                    "format": "sensenova-u1.5-mot",
                    "source_repo": resource_id,
                },
            )
            with (
                mock.patch("transformers.AutoConfig.from_pretrained", return_value=config) as load_config,
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
                mock.patch(
                    "transformers.AutoModel.from_config",
                    side_effect=lambda _config: torch.nn.Linear(2, 2, bias=False),
                ) as build_model,
                mock.patch("sensenova_u1.check_checkpoint_compatibility"),
            ):
                model, loaded_tokenizer = load_model_and_tokenizer(
                    str(checkpoint),
                    dtype=torch.bfloat16,
                    for_offload=True,
                )

        self.assertIs(loaded_tokenizer, tokenizer)
        load_config.assert_called_once_with(resource_id)
        load_tokenizer.assert_called_once_with(resource_id)
        build_model.assert_called_once_with(config)
        torch.testing.assert_close(
            model.weight,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        )

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
