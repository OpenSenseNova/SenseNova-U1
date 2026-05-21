# --------------------------------------------------------
# SenseNovaVL — derived from InternVL (OpenGVLab, MIT).
# Copyright (c) 2023 OpenGVLab. Licensed under MIT.
# Copyright (c) SenseNovaLM contributors. Modifications licensed under Apache-2.0.
# --------------------------------------------------------

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_neo_vit import NEOVisionConfig

logger = logging.get_logger(__name__)


class SenseNovaVLChatConfig(PretrainedConfig):
    """
    Config for SenseNovaVL.
    """

    model_type = "sensenovavl_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        pad2square=False,
        select_layer=-4,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        img_context_token_id=200000,
        img_start_token_id=0,
        image_fold=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        image_gen_loss_weight=0.0,
        ps_version="v1",
        timestep_shift=1,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
        noise_scale_mode="fixed",
        noise_scale_base_image_seq_len=64,
        add_noise_scale_embedding=False,
        noise_scale_max_value=8,
        noise_scale=1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        fm_head_dim=1536,
        fm_head_layers=12,
        fm_head_mlp_ratio=1,
        extra_num_layers_post=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the NEOVisionConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).")

        self.vision_config = NEOVisionConfig(**vision_config)
        self.llm_config = llm_config
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.image_fold = image_fold
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = img_start_token_id
        self.image_gen_loss_weight = image_gen_loss_weight
        self.timestep_shift = timestep_shift
        self.time_schedule = time_schedule
        self.time_shift_type = time_shift_type
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.noise_scale_max_value = noise_scale_max_value
        self.noise_scale = noise_scale
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.extra_num_layers_post = extra_num_layers_post

        logger.info(f"vision_select_layer: {self.select_layer}")
        logger.info(f"image_fold: {self.image_fold}")
        logger.info(f"ps_version: {self.ps_version}")
        logger.info(f"min_dynamic_patch: {self.min_dynamic_patch}")
        logger.info(f"max_dynamic_patch: {self.max_dynamic_patch}")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config
        output["model_type"] = self.__class__.model_type
        output["pad2square"] = self.pad2square
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["image_fold"] = self.image_fold
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["ps_version"] = self.ps_version
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch
        output["img_context_token_id"] = self.img_context_token_id

        return output
