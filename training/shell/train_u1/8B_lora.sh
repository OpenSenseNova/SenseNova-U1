#!/bin/bash
# LoRA fine-tuning launcher for SenseNova-U1 8B (dense + MoT image-gen).
#
# This is a thin variant of ``8B.sh`` that:
#   * freezes every base weight,
#   * inserts low-rank adapters into the MoT image-generation path of the LLM
#     (gen attention + FFN, the Wan/DiffSynth convention),
#   * drops the optimizer/EMA cost accordingly,
#   * uses a smaller dataset + more total steps, tuned for ~50–500-image
#     style fine-tunes (Pixar / Studio Ghibli / etc.).
#
# Single-node (8 GPUs):
#   bash shell/train_u1/8B_lora.sh
#
# Override any env var the same way as ``8B.sh``:
#   mm_data_path=data/pixar_lora/pixar_style_meta.json bash shell/train_u1/8B_lora.sh

set -e
cd "$(dirname "$0")/../.."  # repo root

# ============================ Distributed (torchrun) ============================ #
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============================ Model & data (placeholders — fill in!) ============================ #
export CONFIG_NAME="configs/sensenovavl_qwen3_gen/sensenovau1_8b_mot_sft.py"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/path/to/SenseNova-U1-8B-MoT-SFT"}
export VOCAB_FILE=${VOCAB_FILE:-"/path/to/qwen3/tokenizer"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/qwen3/tokenizer"}
# Point this at the meta JSON produced by tools/prepare_lora_dataset.py
# (relative paths are resolved from the training/ directory).
export mm_data_path=${mm_data_path:-"data/pixar_lora/pixar_style_meta.json"}
export load_optimizer=${load_optimizer:-"model"}

# ============================ Parallelism ============================ #
# Same as the SFT launcher; LoRA does not change topology.
export zero1_size=-1
export wp_size=8
export tp_size=1
export pp_size=1

# ============================ Optimization ============================ #
# Higher LR is fine — only a few million params are being updated and the
# LoRA-B init is zero, so initial gradients are well-behaved.
export SEED=42
export lr=${lr:-1e-4}
export lr_scheduler_type="constant"
export min_lr_ratio=1.0
export mlp_lr_scale=1.0
export weight_decay=0
export grad_accm=1
export total_steps=${total_steps:-5000}
export init_steps=${init_steps:-100}

# ============================ Data / sequence ============================ #
# A style fine-tune does not need a giant packing budget — drop seq_len /
# num_imgs aggressively so a smaller GPU box still fits.
export num_imgs=${num_imgs:-16}
export seq_len=${seq_len:-8192}
export max_sample_tokens=${max_sample_tokens:-8192}
export dataset_replacement=true
export min_num_frame=1
export max_num_frame=128
export dynamic_image_version="native_resolution"
export CONV_STYLE="sensenovalm2-chat-v3"
export down_sample_ratio=0.5
export max_pixels=$((1024 * 1024))
export min_pixels=$((256 * 256))
export max_pixels_gen=$((512 * 512))
export min_pixels_gen=$((256 * 256))
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

# ============================ Freeze / trainable modules ============================ #
# Base model fully frozen — the LoRA injector will unfreeze only its own params.
export mot_random_init=false
export freeze_llm=true
export freeze_backbone=true
export freeze_mlp=true
export unfreeze_mot_gen=false

# ============================ LoRA ============================ #
export lora_enabled=true
export lora_r=${lora_r:-32}
export lora_alpha=${lora_alpha:-32}   # alpha == r -> scale 1.0 (Wan/DiffSynth default)
export lora_dropout=${lora_dropout:-0.0}
# Which generation-path weights to adapt:
#   gen_attn_ffn -> attention + FFN of every LLM layer (Wan standard, default)
#   gen_attn     -> attention only (original-LoRA-paper style)
export lora_target=${lora_target:-"gen_attn_ffn"}
export lora_target_prefixes=${lora_target_prefixes:-"language_model.layers."}

# ============================ Generation / diffusion ============================ #
export time_schedule="standard"
export time_shift_type="exponential"
export time_base_dist="logit_normal"
export base_shift=0.5
export max_shift=1.15
export base_image_seq_len=64
export max_image_seq_len=4096
export noise_scale_mode="resolution"
export noise_scale_base_image_seq_len=64
export add_noise_scale_embedding=true
export noise_scale_max_value=8
export P_mean=-0.8
export P_std=0.8
export cfg_txt_uncond_drop_prob=0.1
export cfg_img_uncond_drop_prob=0
export cfg_txtimg_uncond_drop_prob=0.1
export cfg_is_uncond_drop_independent='false'
# EMA is disabled by the config when lora_enabled=true (the shadow copy of a
# frozen base wastes HBM and step time).
export thinking_method="tag"

# ============================ Understanding ============================ #
export pad_dummy_image_gen='true'
export ce_loss_weight=0.0
export enable_und_loss='false'

# ============================ Job / logging ============================ #
export JOB_NAME=${JOB_NAME:-"sensenovau1_8b_lora_pixar"}
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
# export WANDB_PROJECT="neo_unify"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================ Launch ============================ #
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_sensenovau1.py \
        --config "${CONFIG_NAME}" \
        --launcher torch \
        --seed "${SEED}" \
        --backend nccl
