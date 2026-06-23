# SenseNova-U1 8B (dense + MoT image-gen branch) — SFT config.
#
# Consumed by `train_sensenovau1.py --config`. Runtime-tuned knobs come from
# environment variables (set by `shell/train_u1/8B.sh`); the rest are constants.
import os

from sensenovalm.utils.config_helpers import env_bool


# -----------------------------------------------------------------------------
# Job / cluster
# -----------------------------------------------------------------------------
JOB_NAME = os.environ['JOB_NAME']
print(f'JOB_NAME {JOB_NAME}')


# -----------------------------------------------------------------------------
# Pretrained weights / data paths
# -----------------------------------------------------------------------------
VOCAB_FILE = os.environ.get("VOCAB_FILE", "/path/to/qwen3/tokenizer")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/path/to/qwen3/tokenizer")

MODEL_NAME_OR_PATH = os.environ.get('MODEL_NAME_OR_PATH', None)
LLM_PATH = os.environ.get('LLM_PATH', None)
VIT_PATH = os.environ.get('VIT_PATH', None)
MLP_PATH = os.environ.get('MLP_PATH', None)
MODEL_ONLY_FOLDER = os.environ.get('MODEL_ONLY_FOLDER', None)
mm_data_path = os.environ.get('mm_data_path', None)
print(f'will use vlm model {MODEL_NAME_OR_PATH}')
print(f'will use llm model {LLM_PATH}')
print(f'will use vit model {VIT_PATH}')
print(f'will use mlp model {MLP_PATH}')
print(f'will use data config {mm_data_path}')


# -----------------------------------------------------------------------------
# Parallelism
# -----------------------------------------------------------------------------
zero1_size = int(os.environ['zero1_size'])
wp_size = int(os.environ['wp_size'])
tp_size = int(os.environ['tp_size'])
pp_size = int(os.environ['pp_size'])


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
lr = float(os.environ['lr'])
weight_decay = float(os.environ['weight_decay'])
grad_accm = int(os.environ['grad_accm'])
total_steps = int(os.environ['total_steps'])
init_steps = int(os.environ['init_steps'])
min_lr_ratio = float(os.environ['min_lr_ratio'])
mlp_lr_scale = float(os.environ['mlp_lr_scale'])
fm_modules_lr_scale = float(os.environ.get('fm_modules_lr_scale', 1.0))
mot_gen_lr_scale = float(os.environ.get('mot_gen_lr_scale', 1.0))
lr_scheduler_type = os.environ.get('lr_scheduler_type', 'cosine')
lr_scheduler_offset = int(os.environ.get('lr_scheduler_offset', 0))
load_optimizer = os.environ.get('load_optimizer', None)
ce_loss_weight = float(os.environ.get('ce_loss_weight', 1.0))


# -----------------------------------------------------------------------------
# Data / sequence
# -----------------------------------------------------------------------------
num_imgs = int(os.environ['num_imgs'])
seq_length = int(os.environ['seq_len'])
SEQ_LEN = seq_length

conv_style = os.environ.get('CONV_STYLE', 'Hermes-2')
print(f'will use conv_style {conv_style}')

min_num_frame = int(os.environ.get('min_num_frame', '4'))
max_num_frame = int(os.environ.get('max_num_frame', '24'))

max_pixels = int(os.environ.get('max_pixels', None))
min_pixels = int(os.environ.get('min_pixels', None))
max_pixels_gen = int(os.environ.get('max_pixels_gen', max_pixels))
min_pixels_gen = int(os.environ.get('min_pixels_gen', min_pixels))
dynamic_image_version = os.environ.get('dynamic_image_version', 'native_resolution')
down_sample_ratio = float(os.environ.get('down_sample_ratio', 0.5))
print(f'down_sample_ratio is {down_sample_ratio}')

dataset_replacement = env_bool('dataset_replacement', False)

# LLM-text mixing (kept as a hook; both `train_u1/*.sh` set the weights to 0).
llm_data_config = None
llm_data_weights = float(os.environ.get('LLM_DATA_WEIGHTS', 0.5))
mm_cc_data_weights = float(os.environ.get('MM_CC_DATA_WEIGHTS', 0.0))
print(f'llm_data_weights will be {llm_data_weights}')
print(f'mm_cc_data_weights will be {mm_cc_data_weights}')

# Chat-template / thinking knobs (read out of `gpc.config` by `data/dataset.py`).
add_think_tag = env_bool('add_think_tag', False)
image_token_position_random = os.environ.get('image_token_position_random', 'none')
thinking_method = os.environ.get('thinking_method', 'system')
st_prompt_ratio = float(os.environ.get('st_prompt_ratio', 0.0))
rm_all_think_token = env_bool('rm_all_think_token', False)


# -----------------------------------------------------------------------------
# Freeze / trainable modules
# -----------------------------------------------------------------------------
freeze_llm = env_bool('freeze_llm', False)
freeze_mlp = env_bool('freeze_mlp', False)
freeze_backbone = env_bool('freeze_backbone', False)
train_buffer = env_bool('train_buffer', False)
unfreeze_post_buffer = env_bool('unfreeze_post_buffer', False)
unfreeze_mot_gen = env_bool('unfreeze_mot_gen', False)
mot_random_init = env_bool('mot_random_init', True)
post_layer_num = int(os.environ.get('post_layer_num', 0))


# -----------------------------------------------------------------------------
# LoRA (parameter-efficient fine-tuning of the image-generation path).
#
# When ``lora_enabled=true`` the whole model is frozen and only low-rank
# adapters are trained. By default we follow the Wan/DiffSynth convention:
# adapt the generation-path attention AND FFN of every LLM layer.
#   lora_target=gen_attn      -> attention only ({wq,wk,wv,wo}_mot_gen)
#   lora_target=gen_attn_ffn  -> attention + FFN (default; adds w1/w2/w3)
# See sensenovavl/model/lora.py and README_LORA.md.
# -----------------------------------------------------------------------------
lora_enabled = env_bool('lora_enabled', False)
lora_r = int(os.environ.get('lora_r', 32))
lora_alpha = int(os.environ.get('lora_alpha', 32))  # alpha == r -> scale 1.0 (Wan default)
lora_dropout = float(os.environ.get('lora_dropout', 0.0))
lora_target = os.environ.get('lora_target', 'gen_attn_ffn')
# Comma-separated qualname prefixes; default = the LLM transformer layers.
lora_target_prefixes = tuple(
    p.strip() for p in os.environ.get('lora_target_prefixes', 'language_model.layers.').split(',') if p.strip()
)
_LORA_ATTN_LEAVES = ('wq_mot_gen', 'wk_mot_gen', 'wv_mot_gen', 'wo_mot_gen')
_LORA_FFN_LEAVES = ('w1', 'w2', 'w3')
if lora_target == 'gen_attn':
    lora_target_leaf_names = _LORA_ATTN_LEAVES
elif lora_target == 'gen_attn_ffn':
    lora_target_leaf_names = _LORA_ATTN_LEAVES + _LORA_FFN_LEAVES
else:
    raise ValueError(f"unknown lora_target={lora_target!r}; choose 'gen_attn' or 'gen_attn_ffn'")


# -----------------------------------------------------------------------------
# Generation / flow-matching diffusion
# -----------------------------------------------------------------------------
P_mean = float(os.environ.get('P_mean', 0.0))
P_std = float(os.environ.get('P_std', 1.0))
time_schedule = os.environ.get('time_schedule', 'standard')
time_shift_type = os.environ.get('time_shift_type', 'exponential')
base_shift = float(os.environ.get('base_shift', 0.5))
max_shift = float(os.environ.get('max_shift', 1.15))
base_image_seq_len = int(os.environ.get('base_image_seq_len', 64))
max_image_seq_len = int(os.environ.get('max_image_seq_len', 4096))
noise_scale_mode = os.environ.get('noise_scale_mode', 'fixed')
noise_scale = float(os.environ.get('noise_scale', 1))
noise_scale_base_image_seq_len = int(os.environ.get('noise_scale_base_image_seq_len', 64))
noise_scale_max_value = float(os.environ.get('noise_scale_max_value', 8))
add_noise_scale_embedding = env_bool('add_noise_scale_embedding', False)

# Classifier-free guidance unconditional-drop probabilities (applied per sample).
cfg_txt_uncond_drop_prob = float(os.environ.get('cfg_txt_uncond_drop_prob', 0.1))
cfg_img_uncond_drop_prob = float(os.environ.get('cfg_img_uncond_drop_prob', 0.1))
cfg_txtimg_uncond_drop_prob = float(os.environ.get('cfg_txtimg_uncond_drop_prob', 0))
cfg_is_uncond_drop_independent = env_bool('cfg_is_uncond_drop_independent', True)

# EMA of the unified model.
ema_decay = float(os.environ.get('ema_decay', 0.999))


# -----------------------------------------------------------------------------
# Understanding (CE on text tokens)
# -----------------------------------------------------------------------------
add_gen_tokens_for_und = env_bool('add_gen_tokens_for_und', False)
pad_dummy_image_gen = env_bool('pad_dummy_image_gen', False)
enable_und_loss = env_bool('enable_und_loss', True)
if not freeze_backbone or not freeze_llm:
    assert enable_und_loss


# -----------------------------------------------------------------------------
# Resume
# -----------------------------------------------------------------------------
auto_resume = env_bool('auto_resume', False)
resume_ds = env_bool('resume_ds', False)


# -----------------------------------------------------------------------------
# Model size (LLM backbone)
# -----------------------------------------------------------------------------
VOCAB_SIZE = 151936

# 8B dense
HIDDEN_SIZE = 4096
HEAD_DIM = 128
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 12288 / 4096
EXTRA_NUM_LAYER = 6
EXTRA_NUM_LAYER_POST = post_layer_num
NUM_LAYER = 36 + EXTRA_NUM_LAYER + EXTRA_NUM_LAYER_POST

model_type = "QWEN3_MoEMoT"

if llm_data_config is not None:
    llm_data_config['vocab_file'] = VOCAB_FILE


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------
SAVE_CKPT_FOLDER = f"local:RUN/{JOB_NAME}"
CHECKPOINT_EVERY = 100             # full ckpt save every N train steps
CHECKPOINT_SNAPSHOT_EVERY = 1000   # snapshot ckpt save every N train steps

ckpt = dict(
    enable_save_ckpt=True,
    save_ckpt_folder=SAVE_CKPT_FOLDER,
    load_ckpt_folder=MODEL_ONLY_FOLDER,
    # load_ckpt_info: path = ckpt dir; content = restored states
    # ("model" / "sampler" / "optimizer" / "scheduler" / "all");
    # ckpt_type = format ("internevo" / "llama" / "hf_llama").
    load_ckpt_info=dict(
        path=MODEL_ONLY_FOLDER,
        content=(load_optimizer,),
        ckpt_type="internevo",
    ),
    # When auto_resume=True the trainer reloads the latest snapshot from
    # save_ckpt_folder on restart (so the run survives hardware blips); set
    # resume_ds=True to also restore the data sampler.
    auto_resume=auto_resume,
    resume_ds=resume_ds,
    checkpoint_every=CHECKPOINT_EVERY,
    oss_snapshot_freq=CHECKPOINT_SNAPSHOT_EVERY,
    async_upload=True,
    async_upload_tmp_folder="/dev/shm/sensenovalm_tmp_ckpt/",
)


# -----------------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------------
data = dict(
    type="multimodal_streaming",
    use_packed_ds=True,
    seq_len=SEQ_LEN,
    micro_num=grad_accm,
    micro_bsz=1,
    pack_sample_into_one=False,
    total_steps=total_steps,
    total_epochs=1,
    skip_batches="",
    min_length=50,
    train_folder=None,  # SenseNovaVL drives this via data.meta_path
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    num_workers=8,
    rampup_batch_size="",
    # vision / multimodal
    conv_style=conv_style,
    meta_path=mm_data_path,
    tokenizer_path=TOKENIZER_PATH,
    hf_image_size=512,
    image_size=512,
    force_image_size=512,
    patch_size=16,
    down_sample_ratio=down_sample_ratio,
    pad2square=False,
    use_thumbnail=False,
    group_by_length=False,
    dynamic_image_size=True,
    dynamic_image_version=dynamic_image_version,
    max_pixels=max_pixels,
    min_pixels=min_pixels,
    max_pixels_gen=max_pixels_gen,
    min_pixels_gen=min_pixels_gen,
    min_num_frame=min_num_frame,
    max_num_frame=max_num_frame,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    # packing
    num_images_expected=num_imgs,
    max_packed_tokens=SEQ_LEN,
    max_buffer_size=10,
    log_freq=1000,
    strict_mode=False,
    split_data_chunk=True,
    data_augment=False,
    replacement=dataset_replacement,
    # LLM-text mixing
    llm_data_config=llm_data_config,
    llm_data_weights=llm_data_weights,
    mm_cc_data_weights=mm_cc_data_weights,
    nlp_mm_sampling_fixed_token=True,
    # loss reduction
    loss_reduction='square',
    loss_reduction_all_gather=True,
    use_bos=False,
    use_eos=True,
    # generation
    candidate_resolutions_for_gen=[(256, 256)],
    cfg_txt_uncond_drop_prob=cfg_txt_uncond_drop_prob,
    cfg_img_uncond_drop_prob=cfg_img_uncond_drop_prob,
    cfg_txtimg_uncond_drop_prob=cfg_txtimg_uncond_drop_prob,
    cfg_is_uncond_drop_independent=cfg_is_uncond_drop_independent,
    enabel_und_loss=enable_und_loss,
)


# -----------------------------------------------------------------------------
# Model (LLM backbone + MoT image-gen branch + ViT adapter)
# -----------------------------------------------------------------------------
model = dict(
    # backbone shape
    head_dim=HEAD_DIM,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    hidden_size=HIDDEN_SIZE,
    mlp_ratio=MLP_RATIO,
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYER,
    extra_num_layers=EXTRA_NUM_LAYER,
    extra_num_layers_post=EXTRA_NUM_LAYER_POST,
    max_position_embeddings=262144,
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-6,
    qkv_bias=False,
    rope_base=5000000.0,
    rope_scaling_factor=1.0,
    use_sliding_window=False,
    multiple_of=1,
    qk_interleaved=False,
    attention_type="SWA",
    apply_post_layer_norm=False,
    embed_grad_scale=1,
    embed_split_hidden=True,
    parallel_output=True,
    mlp_layer_fusion=False,
    attention_selective_checkpoint=False,
    num_chunks=1,
    # activation checkpointing fraction: True/False/[0-1]
    checkpoint=1,
    use_flash_attn=True,
    use_cache=False,
    pure_llm=False,
    dtype="torch.bfloat16",
    # pretrained
    model_name_or_path=MODEL_NAME_OR_PATH,
    vision_path=VIT_PATH,
    llm_path=LLM_PATH,
    mlp_path=MLP_PATH,
    # freeze / trainable
    freeze_llm=freeze_llm,
    freeze_mlp=freeze_mlp,
    freeze_backbone=freeze_backbone,
    train_buffer=train_buffer,
    unfreeze_post_buffer=unfreeze_post_buffer,
    unfreeze_mot_gen=unfreeze_mot_gen,
    unfreeze_vit_layers=0,
    unfreeze_lm_head=False,
    # MoT branch + vision adapter
    mot_model=True,
    mot_random_init=mot_random_init,
    vision_select_layer=-1,
    image_fold=None,
    ps_version='v2',
    down_sample_ratio=down_sample_ratio,
    # flow-matching image generation
    timestep_shift=1,
    time_schedule=time_schedule,
    time_shift_type=time_shift_type,
    base_shift=base_shift,
    max_shift=max_shift,
    base_image_seq_len=base_image_seq_len,
    max_image_seq_len=max_image_seq_len,
    noise_scale_mode=noise_scale_mode,
    noise_scale_base_image_seq_len=noise_scale_base_image_seq_len,
    add_noise_scale_embedding=add_noise_scale_embedding,
    noise_scale_max_value=noise_scale_max_value,
    noise_scale=noise_scale,
    P_mean=P_mean,
    P_std=P_std,
    t_eps=0.05,
    fm_head_dim=1536,
    fm_head_layers=2,
    fm_head_mlp_ratio=1,
    # ViT adapter (zero local layers; pretrained ViT weights come from HF).
    vit_cfg=dict(
        num_hidden_layers=0,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        hidden_act='gelu',
        norm_type="layer_norm",
        qkv_bias=True,
        proj_bias=True,
        qk_normalization=False,
        dropout=0,
        drop_path_rate=0.0,
        use_flash_attn=True,
        gradient_checkpointing=True,
        encode_checkpointing=True,
        use_moe=False,
        llm_hidden_size=HIDDEN_SIZE,
        moe_cfg=dict(
            moe_type="GShard_VL",
            num_experts=8,
            num_routed_experts=4,
            use_residual=True,
            num_shared_experts=4,
            shared_expert_intermediate_size=int(768 * 4),
            use_weighted_residual=False,
            routed_expert_jitter=True,
            jitter_epsilon=0.05,
            coef_loss_after_mean=True,
            coef_linear_bias=False,
            capacity_factor=1.2,
            eval_capacity_factor=1.4,
            min_capacity=4,
            noisy_gate_policy="RSample_before",
            moe_intermediate_size=768,
            drop_tokens=True,
            use_rts=False,
            laux_allreduce="all_nodes",
            moe_output_scale=4.0,
            moe_coeff_ratio=0.5,
            split_size=4,
            multiple_of=128,
        ),
    ),
)


# -----------------------------------------------------------------------------
# LoRA config block — consumed by sensenovavl/train/pipeline.py::get_model and
# sensenovalm/checkpoint/checkpoint_manager.py.
# -----------------------------------------------------------------------------
lora = dict(
    enabled=lora_enabled,
    r=lora_r,
    alpha=lora_alpha,
    dropout=lora_dropout,
    target_prefixes=lora_target_prefixes,
    target_leaf_names=lora_target_leaf_names,
    # Every generation-path weight has "mot_gen" in its qualname; requiring it
    # disambiguates the FFN w1/w2/w3 (the frozen understanding FFN reuses those
    # leaf names) and excludes the understanding attention wq/wk/wv/wo.
    require_substrings=('mot_gen',),
    include_sequential_indices=False,
)


# -----------------------------------------------------------------------------
# EMA copy of the model
# -----------------------------------------------------------------------------
# Disabled under LoRA: the EMA shadow copies *all* params, which for a frozen
# base is GBs of HBM and a per-step update that can never change the result.
averaged_model = dict(
    enable=not lora_enabled,
    decay=ema_decay,
    multi_avg_fn="ema",
    use_buffers=False,
)


# -----------------------------------------------------------------------------
# Optimizer / scheduler / loss
# -----------------------------------------------------------------------------
adam = dict(
    lr=lr,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-08,
    adam_beta2_c=0,
    weight_decay=weight_decay,
)

lr_scale = dict(
    vit_layer_decay_rate=1.0,
    moe_layer_decay_rate=1.0,
    vit_woinit_layer_decay_rate=1.0,
    mlp_lr_scale=mlp_lr_scale,
    mot_gen_lr_scale=mot_gen_lr_scale,
    fm_modules_lr_scale=fm_modules_lr_scale,
    moe_wg_lr_scale=1.0,
    moe_coeff_lr_scale=1.0,
)

lr_scheduler = dict(
    total_steps=total_steps,
    init_steps=init_steps,
    warmup_ratio=0.0,
    last_epoch=-1,
    eta_min=min_lr_ratio,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

# FP16 loss-scaling. Hard-pinned to scale=1 since we train in bf16.
grad_scaler = dict(
    fp16=dict(initial_scale=1, min_scale=1, growth_interval=1000),
    growth_factor=1,
    backoff_factor=1,
    max_scale=1,
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    overlap_sync_grad=True,
    overlap_sync_param=False,
    reduce_bucket_size=256 * 1024 * 1024,
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
    image_gen_loss_coef=1.0,
    mtp_loss_coeff=0.1,
)


# -----------------------------------------------------------------------------
# Parallelism
# -----------------------------------------------------------------------------
# zero1   : ZeRO-1 sharding. size<=0 → match DP size; size==1 → disable.
# tensor  : TP mode in {"mtp","msp","fsp","isp"}. We use ISP (intern sequence
#           parallel, decouples TP from SP, composes with weight parallel).
# pipeline: ``interleaved_overlap`` overlaps comm with interleaved PP scheduler.
# weight  : weight parallel; ``overlap=True`` hides allgather/reduce-scatter.
# expert / expert_weight / expert_zero1: same options for MoE experts.
parallel = dict(
    zero1=dict(size=zero1_size, fsdp=False),
    tensor=dict(size=tp_size, mode="isp"),
    pipeline=dict(size=pp_size, interleaved_overlap=True),
    weight=dict(size=wp_size, overlap=True, memory_pool=False),
    expert=dict(size=1),
    expert_zero1=dict(size=1),
    expert_weight=dict(size=1, overlap=True, launch_allgather_before="wo", forward_overlap_per="layer"),
)


# -----------------------------------------------------------------------------
# Monitoring / system flags
# -----------------------------------------------------------------------------
monitor = dict(
    alert=dict(
        enable_feishu_alert=False,
        feishu_alert_address=None,   # feishu webhook URL
        light_monitor_address=None,  # light_monitor heartbeat target
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(queue_max_length=10, interval_step=5),
)
tensorboard = dict(queue_max_length=100)

cudnn_benchmark = False
cudnn_deterministic = False
use_fp32_norm = False
MP_SPAWN = False
