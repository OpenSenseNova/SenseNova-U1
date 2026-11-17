from unittest.mock import patch

import pytest
import torch

from sensenova_u1.utils.checkpoint_loading import (
    load_model_and_tokenizer_for_format,
    resolve_model_runtime_format,
)


def test_resolve_model_runtime_format_keeps_explicit_formats() -> None:
    assert resolve_model_runtime_format("checkpoint.safetensors", "bf16") == "bf16"
    assert resolve_model_runtime_format("checkpoint.safetensors", "int8-convrot") == "int8-convrot"
    assert resolve_model_runtime_format("checkpoint.gguf", "auto") == "gguf"


def test_quantized_runtime_uses_native_loader() -> None:
    with patch("sensenova_u1.utils.quant_runtime.load_quant", return_value=("model", "tokenizer")) as load_quant:
        loaded = load_model_and_tokenizer_for_format(
            "checkpoint.safetensors",
            model_format="int8-convrot",
            dtype=torch.bfloat16,
            device="cuda",
        )

    assert loaded == ("model", "tokenizer")
    load_quant.assert_called_once_with("checkpoint.safetensors", "cuda")


@pytest.mark.parametrize("kwargs", [{"for_offload": True}, {"device_map": "auto"}])
def test_quantized_runtime_rejects_incompatible_placement(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="full-device loading"):
        load_model_and_tokenizer_for_format(
            "checkpoint.safetensors",
            model_format="fp8",
            dtype=torch.bfloat16,
            device="cuda",
            **kwargs,
        )


def test_standard_runtime_keeps_existing_loader_path() -> None:
    with patch(
        "sensenova_u1.utils.checkpoint_loading.load_model_and_tokenizer",
        return_value=("model", "tokenizer"),
    ) as load_model:
        loaded = load_model_and_tokenizer_for_format(
            "checkpoint-dir",
            model_format="bf16",
            dtype=torch.bfloat16,
            device="cuda",
        )

    assert loaded == ("model", "tokenizer")
    load_model.assert_called_once()
