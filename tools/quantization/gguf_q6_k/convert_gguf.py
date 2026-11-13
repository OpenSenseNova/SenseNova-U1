"""Quantize U1.6 linear weights and write a complete GGUF checkpoint."""

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
from convert_utils import finish, prepare_output, tensors
from gguf.quants import quantize


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
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qtype", choices=["Q4_0", "Q6_K", "Q8_0"], required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    partial = prepare_output(a.model_path, a.output)
    qtype = getattr(gguf.GGMLQuantizationType, a.qtype)
    quant = quantize
    if a.qtype == "Q6_K":
        from native_q6 import quantize_q6

        quant = quantize_q6
    writer = gguf.GGUFWriter(partial, "sensenova_u1.6")
    writer.add_name("SenseNova-U1.6-8B-MoT-" + a.qtype)
    writer.add_quantization_version(2)
    writer.add_file_type(int(getattr(gguf.LlamaFileType, "MOSTLY_" + a.qtype)))
    expected, quantized = {}, 0
    for name, tensor in tensors(a.model_path):
        array = tensor.float().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite input: {name}")
        if name == "language_model.lm_head.weight":
            selected = True
        else:
            selected = should_quantize(name)
        if selected:
            dtype = qtype
        else:
            dtype = gguf.GGMLQuantizationType.F32
        if selected:
            if array.ndim != 2 or array.shape[-1] % 256:
                raise ValueError(f"Unaligned linear weight: {name}")
            array = quant(array, qtype)
            quantized += 1
        writer.add_tensor(name, array, raw_dtype=dtype)
        expected[name] = (list(tensor.shape), dtype)
        if len(expected) % 50 == 0:
            print(f"Converted {len(expected)}/1116 tensors", flush=True)
    if (len(expected), quantized) != (1116, 589):
        raise ValueError(f"Unexpected U1.6 coverage: {len(expected)}, {quantized}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(partial)
    actual = {t.name: (list(map(int, t.shape[::-1])), t.tensor_type) for t in reader.tensors}
    if actual != expected or len(reader.tensors) != len(expected):
        raise ValueError("GGUF name/shape/type verification failed")
    del reader
    finish(partial, a.output, a.model_path, format=a.qtype, tensors=len(expected), quantized=quantized)


if __name__ == "__main__":
    main()
