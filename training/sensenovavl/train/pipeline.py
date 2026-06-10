# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import torch

from sensenovalm.core.context import Config
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger
from sensenovavl.model.sensenovavl_moe_chat import (
    SenseNovaVLChatConfig,
    build_pipeline_partition_mot_model,
)

from sensenovavl.utils.checkpoint import (
    load_pretrained_llm,
    load_pretrained_mlp,
    load_pretrained_model,
    load_pretrained_vit,
)

# global llm logger
logger = get_logger(__file__)


def get_model(model_args, data_args):
    if gpc.is_rank_for_log():
        logger.info("Building SenseNovaVLChatConfig...")

    load_hf_vit = False
    load_hf_llm = False
    load_hf_mlp = False
    load_hf_model = False
    if model_args.get("vision_path", None) is not None:
        load_hf_vit = True
    if model_args.get("llm_path", None) is not None:
        load_hf_llm = True
    if model_args.get("mlp_path", None) is not None:
        load_hf_mlp = True
    if model_args.get("model_name_or_path", None) is not None:
        load_hf_model = True

    if load_hf_model is True:
        if gpc.is_rank_for_log():
            logger.info(
                "When the model_path is not empty, the model_path will be loaded in priority. \
                Please set the model_path to None if want to load model from seperate path."
            )

    vit_cfg = model_args.vit_cfg
    vit_cfg.downsample_ratio = data_args.down_sample_ratio
    
    if load_hf_vit or load_hf_model:
        vit_cfg.image_size = data_args.hf_image_size
    else:
        vit_cfg.image_size = data_args.image_size

    vision_select_layer = model_args.get("vision_select_layer", -1)
    image_fold = model_args.get("image_fold", None)
    ps_version = model_args.get("ps_version", "V2")
    moe_location = model_args.get("moe_location", "")
    llm_config = model_args

    if "use_moe" not in vit_cfg:
        vit_cfg.use_moe = False
    moe_kwargs = model_args.moe_kwargs
    if model_args.use_moe:
        assert moe_location in ["vision", "llm"], (
            "moe_location is not specified. if you want to use moe, "
            "please set moe_location to 'vision' or 'llm', else set num_experts to 1."
        )
        if moe_location == "vision":
            vit_cfg.use_moe = True
            vit_cfg.moe_cfg = moe_kwargs
            vit_cfg.moe_layer_kwargs = model_args.get("moe_layer_kwargs", {})
            vit_cfg.moe_cfg["num_routed_experts"] = moe_kwargs.top_k

    if gpc.is_rank_for_log():
        logger.info(f"llm_config: {llm_config}")
        logger.info(f"vit_config: {vit_cfg}")

    llm_config = Config(llm_config)

    sensenovavl_chat_config = SenseNovaVLChatConfig(
        vision_config=vit_cfg,
        llm_config=llm_config,
        downsample_ratio=data_args.down_sample_ratio,
        select_layer=vision_select_layer,
        image_fold=image_fold,
        ps_version=ps_version,
        img_context_token_id=data_args.get("img_context_token_id", None),
        img_start_token_id=data_args.get("img_start_token_id", None),
        force_image_size=data_args.force_image_size,
        image_gen_loss_weight=gpc.config.loss.get("image_gen_loss_coef", 0.0),
        timestep_shift=model_args.timestep_shift,
        time_schedule=model_args.time_schedule,
        time_shift_type=model_args.time_shift_type,
        base_shift=model_args.base_shift,
        max_shift=model_args.max_shift,
        base_image_seq_len=model_args.base_image_seq_len,
        max_image_seq_len=model_args.max_image_seq_len,
        noise_scale_mode=model_args.noise_scale_mode,
        noise_scale_base_image_seq_len=model_args.noise_scale_base_image_seq_len,
        add_noise_scale_embedding=model_args.add_noise_scale_embedding,
        noise_scale_max_value=model_args.noise_scale_max_value,
        noise_scale=model_args.noise_scale,
        P_mean=model_args.P_mean,
        P_std=model_args.P_std,
        t_eps=model_args.t_eps,
        fm_head_dim=model_args.fm_head_dim,
        fm_head_layers=model_args.fm_head_layers,
        fm_head_mlp_ratio=model_args.fm_head_mlp_ratio,
        extra_num_layers_post=model_args.extra_num_layers_post,
    )

    if gpc.is_rank_for_log():
        logger.info("Building SenseNovaVLChatMoTModel...")
    model = build_pipeline_partition_mot_model(config=sensenovavl_chat_config, vision_model=None, language_model=None)

    # wait model building completed for all ranks
    torch.distributed.barrier()

    if load_hf_model:
        if gpc.is_rank_for_log():
            logger.info("Loading pretrained whole model...")

        load_pretrained_model(model, vit_cfg, llm_config, model_args.model_name_or_path)

    if load_hf_vit and hasattr(model, "vision_model"):
        if gpc.is_rank_for_log():
            logger.info("Loading pretrained ViT...")
        message = load_pretrained_vit(model.vision_model, model_args=vit_cfg, model_path=model_args.vision_path)
        if gpc.is_rank_for_log():
            logger.info(message)

    if load_hf_llm and hasattr(model, "language_model"):
        if gpc.is_rank_for_log():
            logger.info("Loading pretrained LLM...")

        message = load_pretrained_llm(model.language_model, llm_config, model_args.llm_path)
        if gpc.is_rank_for_log():
            logger.info(message)

    if load_hf_mlp and hasattr(model, "mlp1"):
        if gpc.is_rank_for_log():
            logger.info("Loading pretrained MLP...")
        message = load_pretrained_mlp(model.mlp1, model_args.mlp_path)
        if gpc.is_rank_for_log():
            logger.info(message)

    model.config.force_image_size = data_args.force_image_size
    model.config.vision_config.image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // data_args.patch_size) ** 2 * (data_args.down_sample_ratio**2)
    )

    if hasattr(model, "vision_model"):
        model.vision_model.gradient_checkpointing = vit_cfg.gradient_checkpointing
        model.vision_model.encoder.gradient_checkpointing = vit_cfg.encode_checkpointing

    # NOTE: postlayer ---- #
    if model_args.train_buffer:
        logger.info(f"Only train buffer with extra {gpc.config.model.extra_num_layers} layers")
        for name, param in model.named_parameters():
            parts = name.split('.')
            if (
                "_h" in name
                or "_w" in name
                or "_hw" in name
                or "vision_model" in name
                or (len(parts) > 2 and parts[2].isdigit() and int(parts[2]) < gpc.config.model.extra_num_layers)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
         logger.info(f"Train the whole model with {gpc.config.model.num_layers} layers")


    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        if hasattr(model, "vision_model"):
            model.vision_model = model.vision_model.eval()
            _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        if hasattr(model, "language_model"):
            model.language_model = model.language_model.eval()
            _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        if hasattr(model, "language_model"):
            model.language_model.output.requires_grad = True

    if model_args.freeze_mlp:
        if hasattr(model, "mlp1"):
            _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        if hasattr(model, "vision_model"):
            layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
            for k, v in layers.named_parameters():
                logger.info(f"Unfreezing ViT layer: {k}")
                v.requires_grad = True

        logger.info("Finished Initializing Model.")

    if getattr(model_args, "unfreeze_post_buffer", False):
        assert model_args.extra_num_layers_post > 0
        for name, param in model.named_parameters():
            parts = name.split('.')
            if (
                (len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= model_args.num_layers-model_args.extra_num_layers_post)
            ):
                param.requires_grad = True

    if getattr(model_args, "unfreeze_mot_gen", False):
        logger.info(f"Unfreeze generation branch in mot model")
        for name, param in model.named_parameters():
            if (
                "mot_gen" in name
            ):
                param.requires_grad = True

    # --- LoRA injection ------------------------------------------------------
    # Runs last so it overrides every prior freeze/unfreeze decision: when
    # ``lora.enabled`` is true we freeze the entire model and only add LoRA
    # adapters under ``fm_modules.*`` as trainable. See
    # ``sensenovavl/model/lora.py`` for the side-branch implementation.
    lora_cfg = gpc.config.get("lora", None) if hasattr(gpc, "config") else None
    if lora_cfg is not None and lora_cfg.get("enabled", False):
        from sensenovavl.model.lora import count_trainable_parameters, inject_lora

        r = int(lora_cfg.get("r", 8))
        alpha = int(lora_cfg.get("alpha", 16))
        dropout = float(lora_cfg.get("dropout", 0.0))
        target_prefixes = tuple(lora_cfg.get("target_prefixes", ("fm_modules.",)))
        target_leaf_names = tuple(
            lora_cfg.get(
                "target_leaf_names",
                ("qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj", "fc1", "fc2"),
            )
        )
        include_seq_idx = bool(lora_cfg.get("include_sequential_indices", True))

        if gpc.is_rank_for_log():
            logger.info(
                f"[LoRA] enabling rank={r} alpha={alpha} dropout={dropout} "
                f"prefixes={list(target_prefixes)} leaves={list(target_leaf_names)}"
            )
        wrapped, n_lora, n_frozen = inject_lora(
            model,
            target_prefixes=target_prefixes,
            target_leaf_names=target_leaf_names,
            include_sequential_indices=include_seq_idx,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        trainable, total = count_trainable_parameters(model)
        if gpc.is_rank_for_log():
            logger.info(
                f"[LoRA] wrapped {len(wrapped)} Linear layers; "
                f"trainable={trainable:,} / total={total:,} "
                f"({100.0 * trainable / max(total, 1):.4f}%)."
            )
            # Show the first few wrapped paths so users can sanity-check that
            # they hit fm_modules and not, e.g., the LLM by accident.
            preview = wrapped[:8] + (["..."] if len(wrapped) > 8 else [])
            logger.info(f"[LoRA] sample wrapped paths: {preview}")

    return model
