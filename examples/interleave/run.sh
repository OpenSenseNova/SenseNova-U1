#!/usr/bin/env bash


repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

model_path=${MODEL_PATH:?Set MODEL_PATH to a local checkpoint path or Hugging Face model ID}
example_dir=${repo_root}/examples/interleave
output_dir=${OUTPUT_DIR:-${example_dir}/output}


# 1) Single sample, text prompt only.
#    Select the output pixel bucket and aspect ratio.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
    --bucket 4m --aspect_ratio 16:9 \
    --output_dir "${output_dir}/text" \
    --stem "demo_text" \
    --profile

# 2) Single sample, text prompt + one input image.
#    Each '<image>' placeholder in the prompt binds to one --image path,
#    in order. With no CLI size options, smart_resize derives the size
#    from the first input image.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --prompt "<image>\n图文交错生成小猫游览故宫的场景" \
    --image "${example_dir}/data/images/image0.jpg" \
    --output_dir "${output_dir}/text_image" \
    --stem "demo_text_image" \
    --profile

# 3) Batch samples. Relative image paths resolve from the repository root.
#    A JSONL sample can override the selected size with 'width' and 'height'.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --jsonl "${example_dir}/data/samples.jsonl" \
    --image_root "${repo_root}" \
    --bucket 4m --aspect_ratio 16:9 \
    --output_dir "${output_dir}/jsonl" \
    --profile

# VBVR-Bench image samples require multi-step visual generation from an input image.
# Note: VBVR-Bench runs on "no-think" mode.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --jsonl "${example_dir}/data/samples_reasoning.jsonl" \
    --image_root "${repo_root}" \
    --output_dir "${output_dir}/reasoning" \
    --no-think_mode \
    --profile
