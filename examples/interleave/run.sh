#!/usr/bin/env bash


repo_root=/mnt/afs/tongwenwen1/workspace/neo/SenseNova-U1

model_path=${MODEL_PATH:-/mnt/afs/wupenghao/workspace/Neo_plus_train/RUN/d20260415_Neo_unify_9B_mot_SFT_fromMT84K_ce01_res2k_lr2e-5_wd0_acc1_img128_seq27648_zs-1_wp16_sp1_gpu512/hf_step9000_ema_rl_ocr_penalty_joint_3200}
example_dir=${repo_root}/examples/interleave
output_dir=${OUTPUT_DIR:-${example_dir}/output}


# 1) Single sample, text prompt only.
#    Output resolution comes from --resolution (default 16:9 -> 2048x1152).

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
    --output_dir "${output_dir}/text" \
    --stem "demo_text" \
    --profile

# 2) Single sample, text prompt + one input image.
#    Each '<image>' placeholder in the prompt binds to one --image path,
#    in order. Output resolution follows the first input image
#    (via smart_resize), ignoring --resolution/--width/--height.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --prompt "<image>\n图文交错生成小猫游览故宫的场景" \
    --image "${example_dir}/data/images/image0.jpg" \
    --output_dir "${output_dir}/text_image" \
    --stem "demo_text_image" \
    --profile

# 3) Each line in the JSONL is one sample:
#    {"prompt": "...", "image": ["images/a.jpg", ...],
#     "width": 2048, "height": 1152, "seed": 42, "think_mode": true}
#    Relative 'image' paths are resolved against --image_root; absolute
#    paths are used as-is. If 'image' is set, the output size follows
#    the first input image; 'width'/'height' only apply to text-only samples.

python "${example_dir}/inference.py" \
    --model_path "${model_path}" \
    --jsonl "${example_dir}/data/sample.jsonl" \
    --image_root "${example_dir}/data/images" \
    --output_dir "${output_dir}/jsonl" \
    --profile
