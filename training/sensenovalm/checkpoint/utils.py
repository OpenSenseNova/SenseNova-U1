#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-
from typing import List

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


def get_shard_state_dict(shard_model):
    """
    Only used for FSDP module saving.
    It's a warper of model.state_dict() and with the context of 'FSDP.state_dict_type', the sharded parameter
    (saved as model.flat_param_xx in sharded FSDP module) will be gathered at every gpu.
    'offload_to_cpu' means that the model states are to be offloaded to cpu chunk by chunk, avoiding OOM in gpu

    """

    # FSDP model can only save with sharded shape SHARDED_STATE_DICT when set use_orig_params=True
    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        shard_states = shard_model.state_dict()

    return shard_states


def get_non_moe_state_dict(full_state_dict):
    """
    Get the state dict of the non-moe layers
    """
    for key in list(full_state_dict.keys()):
        if "expert" in key and "moe_layer.gate" not in key:
            full_state_dict.pop(key)

    return full_state_dict

def obtain_not_saved_mtp_state(state, missing_keys: List[str]):
    mtp_missing_state = {}
    mtp_not_loaded_keys = []
    is_mtp_layer_0_missing = False
    for missing_key in missing_keys:
        if "mtp_layers" in missing_key:
            missing_mtp_layer_idx = int(missing_key.split("mtp_layers.")[1].split(".")[0])
            if missing_mtp_layer_idx == 0 or is_mtp_layer_0_missing:
                if (
                    "norm_before_output" not in missing_key
                    and "norm_after_embedding" not in missing_key
                    and "proj" not in missing_key
                ):
                    # load from main last layer
                    main_last_layer_key = missing_key.replace(
                        f"mtp.mtp_layers.{missing_mtp_layer_idx}.layer", f"layers.{gpc.config.model.num_layers-1}"
                    )
                    if main_last_layer_key == missing_key:
                        main_last_layer_key = missing_key.replace(
                            f"mtp.mtp_layers.{missing_mtp_layer_idx}", f"layers.{gpc.config.model.num_layers-1}"
                        )
                    mtp_missing_state[missing_key] = state[main_last_layer_key]
                is_mtp_layer_0_missing = True
            else:
                # load from previous mtp layer
                previous_mtp_layer_key = missing_key.replace(
                    f"mtp_layers.{missing_mtp_layer_idx}", f"mtp_layers.{missing_mtp_layer_idx-1}"
                )
                mtp_missing_state[missing_key] = state[previous_mtp_layer_key]
            mtp_not_loaded_keys.append(missing_key)
    return mtp_missing_state, mtp_not_loaded_keys


def load_shard_state_dict(shard_model, shard_state, **kwargs):
    """
    Only used for FSDP module loading.

    """

    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        missing_k, unexpected_keys = shard_model.load_state_dict(shard_state, kwargs)

    return (missing_k, unexpected_keys)


def get_model_topology(model):
    """
    Returns:
        {
            '{name}': {'dim': int}
        }
        where name is the name of the module, and all parameters under this module are
        concatenated along the dimension 'dim'.
    """
    topos = {}
    for name, module in model.named_modules():  # pylint: disable=W0612
        # TODO: If it does not meet these conditions, it is shared between various tp/dp, and it is necessary to assert
        # that they are consistent.
        # In order to be compatible with CI, this function will not be deleted for now.
        pass
    return topos


def process_load_info(load_info):
    load_content_str = ""
    load_ckpt_folder = load_info["path"]
    load_content = load_info["content"]
    if gpc.is_rank_for_log():
        logger.info(f"Try load_ckpt_folder: {load_ckpt_folder}")

    return load_content_str, load_ckpt_folder, load_content
