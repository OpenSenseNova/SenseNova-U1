import torch
import torch.nn as nn
from safetensors.torch import safe_open


def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha


def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
    strength: float = 1.0,
):
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    matched = 0
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_state_dict:
            if lora_up_name not in lora_state_dict:
                raise RuntimeError(f"LoRA is missing paired tensor {lora_up_name!r}.")
            lora_down = lora_state_dict[lora_down_name].to(dtype=torch.float32)
            lora_up = lora_state_dict[lora_up_name].to(dtype=torch.float32)
            rank = lora_down.shape[0]
            lora_alpha = float(lora_state_dict.get(lora_alpha_name, rank))
            scaling_factor = float(strength) * lora_alpha / rank
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down).to(value.device)
            with torch.no_grad():
                value.add_(delta_W.to(dtype=value.dtype))
            matched += 1
    if matched == 0:
        raise RuntimeError("LoRA did not match any model weights.")
    return model


def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
    strength: float = 1.0,
):
    lora_state_dict = {}
    with safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    model = load_and_merge_lora_weight(
        model,
        lora_state_dict,
        lora_down_key,
        lora_up_key,
        strength,
    )
    return model
