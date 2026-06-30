# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import json
from typing import Dict, Tuple

import torch

from sensenovalm.core.context.parallel_context import ParallelMode
from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.model.modules.utils import is_moe_param
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


def split_params_into_different_groups_for_optimizer(
    param_groups: Tuple[Dict],
) -> Tuple[Dict]:
    """Split parameters into different groups for optimizer

    Args:
        param_groups (Tuple[Dict]): The list of parameter groups to split
        Input Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>> )

    Returns:
        Tuple[Dict]: list of params groups for optimizer
        Output Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>>     {'name': 'embed_head', 'params': [tensor], 'weight_decay' :xxx},
        >>>     {'name': 'fp32', 'params': [tensor], 'weight_decay' :xxx},
        >>> )
    """

    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    new_groups = {}
    # create new groups for fp32 parameter group
    new_groups["fp32"] = {"name": "fp32", "params": [], "optimizer_mode": ParallelMode.ZERO1}

    if gpc.config.model.use_moe:
        for key in gpc.expert_parallel_group_names:
            new_groups[key] = {"name": key, "moe": True, "params": [], "optimizer_mode": ParallelMode.EXPERT_ZERO1}

    for pgroup in param_groups:
        # copy attribute from origin group, we assume the input param_groups only
        # have one group, so the attribute will not be copyed multiple times.
        for ori_key in pgroup.keys():
            if ori_key not in ("name", "params"):
                for _, group in new_groups.items():
                    group[ori_key] = pgroup[ori_key]
        # assign param
        origin_params = []
        for param in pgroup["params"]:
            # moe param means MoE is enabled
            if is_moe_param(param):
                new_groups[param.group_name]["params"].append(param)
            elif param.dtype == torch.float32 and gpc.config.model.dtype != torch.float32:
                new_groups["fp32"]["params"].append(param)
            else:
                origin_params.append(param)

        # default param group, which is the first group in the param groups
        pgroup["params"] = origin_params
        pgroup["optimizer_mode"] = ParallelMode.ZERO1

    # param groups may contain empty groups, such as fp32
    param_groups.extend(new_groups.values())

    return tuple(param_groups)


def create_param_groups(model, weight_decay, lr, lr_scale, max_group_size=None):

    model_parameters = get_param_group(model.model, weight_decay, lr, lr_scale, max_group_size)

    return model_parameters


def get_param_group(model, weight_decay, learning_rate, lr_scale, max_group_size):
    # old :
    # model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # new
    parameter_groups = {}


    vit_layer_decay_rate = float(lr_scale.get("vit_layer_decay_rate", 1.0))
    mlp_lr_scale = float(lr_scale.get("mlp_lr_scale", 1.0))
    moe_wg_lr_scale = float(lr_scale.get("moe_wg_lr_scale", 1.0))
    moe_coeff_lr_scale = float(lr_scale.get("moe_coeff_lr_scale", 10.0))
    moe_expert_decay_rate = float(lr_scale.get("moe_layer_decay_rate", 10.0))
    fm_modules_lr_scale = float(lr_scale.get("fm_modules_lr_scale", 1.0))

    all_params_wd = gpc.config.get("all_params_wd", False)

    def get_scale(cls):
        if cls in ["vit", "vit_tp"]:
            scale = vit_layer_decay_rate
        elif cls in ["mlp", "mlp_tp"]:
            scale = mlp_lr_scale
        elif cls == "moe_expert":
            scale = moe_expert_decay_rate
        elif cls == "moe_gate":
            scale = moe_wg_lr_scale
        elif cls == "moe_coeff":
            scale = moe_coeff_lr_scale
        elif cls == "fm_modules":
            scale = fm_modules_lr_scale
        else:
            assert cls in ["llm", "llm_tp", "other"]
            scale = 1.0
        return scale

    cls_list = [
        "vit",
        "vit_tp",
        "mlp",
        "mlp_tp",
        "moe_gate",
        "moe_coeff",
        "moe_expert",
        "llm",
        "llm_tp",
        "other",
        "fm_modules",
    ]

    group_name_list = ["no_decay", "decay"]

    # 注意所有

    for cls in cls_list:
        for group_name in group_name_list:
            wd = weight_decay if group_name == "decay" else 0.0
            ls = get_scale(cls)

            group_name = "%s_%s" % (cls, group_name)
            assert group_name not in parameter_groups
            parameter_groups[group_name] = {
                "params": [],
                "param_names": [],
                "group_name": group_name,
                "name": group_name,
                "weight_decay": weight_decay if all_params_wd else wd,
                "lr_scale": ls,
                "lr": learning_rate * ls,
                "optimizer_mode": ParallelMode.ZERO1,
            }
            if cls == "moe_expert" and gpc.config.model.use_moe:
                parameter_groups[group_name]["moe"] = True
                parameter_groups[group_name]["optimizer_mode"] = ParallelMode.EXPERT_ZERO1

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name.endswith("wg.weight")
            or name.endswith("coefficient.weight")
        ):
            group_name = "no_decay"
            this_weight_decay = weight_decay if all_params_wd else 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        cls = param_classification(name)
        group_name = "%s_%s" % (cls, group_name)
        # if "weight_decay" not in parameter_groups[group_name]:

        scale = get_scale(cls)
        parameter_groups[group_name].update(
            {
                "weight_decay": this_weight_decay,
                "lr_scale": scale,
                "lr": scale * learning_rate,
            }
        )
        parameter_groups[group_name]["params"].append(param)
        parameter_groups[group_name]["param_names"].append(name)

    rank = torch.distributed.get_rank()
    if rank == 0:
        to_display = {}
        for key in parameter_groups:  # pylint: disable=C0206
            to_display[key] = {
                "param_names": parameter_groups[key]["param_names"],
                "lr_scale": parameter_groups[key]["lr_scale"],
                "lr": parameter_groups[key]["lr"],
                "weight_decay": parameter_groups[key]["weight_decay"],
            }
        print("Param groups = %s" % json.dumps(to_display, indent=2))

    if max_group_size is not None:
        splited_parameter_groups = []
        for _, v in parameter_groups.items():
            cur_group = []
            all_groups = []
            size_of_cur_group = 0
            for param in v["params"]:
                if size_of_cur_group + param.numel() <= max_group_size:
                    cur_group.append(param)
                    size_of_cur_group += param.numel()
                else:
                    all_groups.append(cur_group)
                    cur_group = [param]
                    size_of_cur_group = param.numel()
            if cur_group:
                all_groups.append(cur_group)
            for group in all_groups:
                new_dict = {}
                for key, val in v.items():
                    if key != "params":
                        new_dict[key] = val
                new_dict["params"] = group
                splited_parameter_groups.append(new_dict)
        optimizer_grouped_parameters = splited_parameter_groups
    else:
        optimizer_grouped_parameters = list(parameter_groups.values())

    return optimizer_grouped_parameters


def param_classification(name):
    if name.startswith("sensenovavl."):
        name = name[len("sensenovavl.") :]

    # LoRA adapters are plain nn.Linear replicated across TP/WP ranks (tagged
    # IS_REPLICA_ZERO_PARALLEL). They must land in a replica param group like
    # ``fm_modules`` — never the weight-sharded ``llm_tp`` group their qualname
    # prefix would otherwise select, which would reduce their grads wrongly.
    if "lora_A" in name or "lora_B" in name:
        return "fm_modules"

    if name.endswith("gate.wg.weight"):
        return "moe_gate"

    elif "mlp.coefficient" in name or "feed_forward.coefficient" in name:
        return "moe_coeff"
    elif "moe_layer.experts" in name:
        return "moe_expert"

    elif name.startswith("vision_model."):
        if "attn.qkv" in name or "attn.proj" in name:
            return "vit_tp"
        else:
            return "vit"
    elif name.startswith("mlp"):
        assert name.startswith("mlp1")
        if name.startswith("mlp1.1") or name.startswith("mlp1.3"):
            return "mlp_tp"
        else:
            return "mlp"
    elif name.startswith("language_model."):
        if (
            "tok_embeddings" in name
            or "output" in name
            or "attention.wqkv" in name
            or "attention.wo" in name
            or "feed_forward" in name
        ):
            return "llm_tp"
        else:
            return "llm"
    elif 'fm_modules' in name:
        return 'fm_modules'
    else:
        return "other"


def split_params_into_different_moe_groups_for_optimizer(
    param_groups: Tuple[Dict], max_group_size=178956971
) -> Tuple[Dict]:
    """Split parameters into different MoE groups for optimizer

    Args:
        param_groups (Tuple[Dict]):
            The list of parameter groups to split

    Returns:
        Tuple[Dict]:
        list of MoE/non-MoE groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # gather all data parallel group names
    data_parallel_group_names = set()
    for param_group in param_groups:
        for param in param_group["params"]:
            if is_moe_param(param):
                data_parallel_group_names.add(param.group_name)
    data_parallel_group_names = list(data_parallel_group_names)
    group_moe = {}
    # Create the param MoE groups, leave param assign to next step
    for param_group in param_groups:
        group_moe[param_group["name"]] = {}
        for key in data_parallel_group_names:
            group_moe[param_group["name"]][key] = {}
            group_moe[param_group["name"]][key]["name"] = key
            group_moe[param_group["name"]][key]["moe"] = True
            group_moe[param_group["name"]][key]["optimizer_mode"] = ParallelMode.EXPERT_ZERO1
            for ori_key in param_group.keys():
                if ori_key != "name":
                    if ori_key == "params":
                        group_moe[param_group["name"]][key][ori_key] = []
                    else:
                        group_moe[param_group["name"]][key][ori_key] = param_group[ori_key]
    # Assign param
    for param_group in param_groups:
        new_params = []
        for param in param_group["params"]:
            if is_moe_param(param):
                group_moe[param_group["name"]][param.group_name]["params"].append(param)
            else:
                new_params.append(param)
        param_group["params"] = new_params

    for param_group in param_groups:
        param_group["optimizer_mode"] = ParallelMode.ZERO1

    moe_group_num = 0
    # Flatten the moe groups
    if max_group_size is not None:
        for _, v in group_moe.items():
            for _, v1 in v.items():
                cur_group = []
                all_groups = []
                size_of_cur_group = 0
                for param in v1["params"]:
                    if size_of_cur_group + param.numel() <= max_group_size:
                        cur_group.append(param)
                        size_of_cur_group += param.numel()
                    else:
                        all_groups.append(cur_group)
                        cur_group = [param]
                        size_of_cur_group = param.numel()
                if cur_group:
                    all_groups.append(cur_group)
                for group in all_groups:
                    new_dict = {}
                    for key, val in v1.items():
                        if key != "params":
                            new_dict[key] = val
                    new_dict["params"] = group
                    param_groups.append(new_dict)
                    moe_group_num += 1
    else:
        for _, v in group_moe.items():
            for _, v1 in v.items():
                param_groups.append(v1)
                moe_group_num += 1

    if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        _moe_group_num = torch.tensor(moe_group_num, dtype=torch.int, device=get_current_device())
        moe_location_is_llm = gpc.config.model.get("moe_location") == "llm"

        if gpc.config.model.use_moe:
            ranks_in_group = gpc.get_ranks_in_group(ParallelMode.PIPELINE)
            local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
            source_rank = ranks_in_group[1] if moe_location_is_llm else ranks_in_group[0]

            torch.distributed.broadcast(
                _moe_group_num,
                source_rank,
                gpc.get_group(ParallelMode.PIPELINE),
            )
            moe_group_name = f"moe_ep_size_{gpc.expert_parallel_size}"

            # vision model only in pp 0
            # for moe in vision model, only pp 0 has the moe group
            # for moe in llm model, pp > 0 has the moe group
            if (moe_location_is_llm and local_rank == 0) or (not moe_location_is_llm and local_rank > 0):
                _param_group = {
                    "name": moe_group_name,
                    "group_name": moe_group_name,
                    "lr": 0,
                    "lr_scale": 0,
                    "weight_decay": 0,
                    "moe": True,
                    "optimizer_mode": ParallelMode.EXPERT_ZERO1,
                    "params": [],
                }
                for _ in range(_moe_group_num.item()):
                    param_groups.append(_param_group)

    return tuple(param_groups)


def timeout_input(printout, default, timeout=None, interactive=True):
    if not interactive:
        return default
    import select
    import sys

    if gpc.is_rank_for_log():
        logger.info(printout)

    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        msg = sys.stdin.readline().strip()
        return default if len(msg) == 0 else msg
    else:
        return default
