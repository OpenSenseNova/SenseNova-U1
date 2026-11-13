"""Read original EMA shards, quantize U1.6 layers, save weights and scales."""

import argparse
import json
from pathlib import Path

import torch
from convert_utils import finish, prepare_output, tensors
from quant import quantize_weight
from safetensors.torch import save_file


def should_quantize(name):
    """Select Attention and MLP linear weights by their layer names."""
    # Example: language_model.model.layers.0.self_attn.q_proj.weight
    parts = name.split(".")
    if len(parts) != 7:
        return False

    if parts[0] != "language_model" or parts[1] != "model" or parts[2] != "layers":
        return False

    layer_index = parts[3]
    module_name = parts[4]
    projection_name = parts[5]
    parameter_name = parts[6]

    if not layer_index.isdecimal():
        return False
    if parameter_name != "weight":
        return False

    if module_name == "self_attn":
        # Understanding branch and image generation branch (names containing _mot_gen).
        return projection_name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_proj_mot_gen",
            "k_proj_mot_gen",
            "v_proj_mot_gen",
            "o_proj_mot_gen",
        ]
    elif module_name == "mlp" or module_name == "mlp_mot_gen":
        return projection_name in ["gate_proj", "up_proj", "down_proj"]
    else:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Original EMA directory containing model.safetensors.index.json")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not (Path(a.model_path) / "model.safetensors.index.json").is_file():
        p.error("--model-path must be the original EMA shard directory")
    torch.set_num_threads(8)
    partial = prepare_output(a.model_path, a.output)
    state, layers = {}, {}
    source_count = 0
    for name, weight in tensors(a.model_path):
        source_count += 1
        if weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
            raise ValueError(f"Expected finite BF16 input: {name}")
        if should_quantize(name):
            q, scale, extra = quantize_weight(weight.to(a.device))
            if not torch.isfinite(q.float()).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"Invalid quantized tensor: {name}")
            state[name], state[name + "_scale"] = q.cpu(), scale.cpu()
            layers[name[:-7]] = extra
            print(f"Quantized {len(layers)}/588: {name}", flush=True)
            del q, scale
        else:
            state[name] = weight
    if (source_count, len(layers)) != (1116, 588):
        raise ValueError(f"Unexpected U1.6 coverage: {source_count}, {len(layers)}")
    first = state[next(iter(layers)) + ".weight"]
    fmt = "fp8" if first.dtype == torch.float8_e4m3fn else "int8-convrot"
    config = dict(schema="sensenova-w8a8-v2", format=fmt, granularity="per_output_channel", layers=layers)
    save_file(state, str(partial), metadata={"format": "pt", "quantization": json.dumps(config)})
    (Path(a.output).parent / "quantization_config.json").write_text(json.dumps(config, indent=2) + "\n")
    finish(partial, a.output, a.model_path, format=fmt, tensors=source_count, quantized=len(layers))


if __name__ == "__main__":
    main()
