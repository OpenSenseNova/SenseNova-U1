# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Uses `LabelSmoother` from HuggingFace transformers (HuggingFace Inc.,
# Apache-2.0) for per-token loss accounting.
import os
import time

import torch
import torch.distributed as dist
from transformers.trainer_pt_utils import LabelSmoother

# isort: off
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.execution_time import execution_time_collecter as etc
from sensenovalm.utils.megatron_timers import megatron_timer as timer
from sensenovalm.utils.timeout import llm_timeout


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ci_loss_list = []


def all_gather_moe_metrics_pp(
    input_: torch.Tensor,
    first_dense_layers: int = 0,
    gather_dim: int = 1,
):
    world_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_group = gpc.get_group(ParallelMode.PIPELINE)
    if world_size <= 1:
        return input_

    pad_size = first_dense_layers
    if first_dense_layers > 0 and gpc.is_pipeline_first_stage():
        shape = list(input_.shape)
        shape[gather_dim] += pad_size
        padded_input = torch.zeros(shape, dtype=input_.dtype, device=input_.device)
        if gather_dim == 1:
            padded_input[:, pad_size:] = input_
        elif gather_dim == 2:
            padded_input[:, :, pad_size:] = input_
    else:
        padded_input = input_

    gather_list = [torch.empty_like(padded_input) for _ in range(world_size)]
    dist.all_gather(gather_list, padded_input, group=pipeline_group)
    output = torch.cat(gather_list, dim=gather_dim).contiguous()
    if gather_dim == 1:
        output = output[:, pad_size:]
    elif gather_dim == 2:
        output = output[:, :, pad_size:]

    return output


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    start_time,
    very_begining_time,
    loss,
    mtp_loss,
    moe_loss,
    moe_z_loss,
    moe_coef_loss,
    boi_loss,
    image_gen_loss,
    losses_for_log_only,
    grad_norm,
    metric,
    num_padding_tokens,
    async_hook,
    **kwargs,
):
    """
    Print some training metrics of current batch.
    """

    moe_monitor_cfg = gpc.config.moe_monitor
    num_padding_tokens_local = num_padding_tokens
    timer.store_last_timers()
    if loss is not None:
        loss_avg = loss.clone()
    else:
        loss_avg = 0.0

    # moe monitor
    if gpc.config.model.use_moe:
        assert moe_loss is not None, f"moe_loss should not be None when training {gpc.config.model_type}. "

        if success_update:
            if gpc.config.model.use_moe and batch_count % moe_monitor_cfg.get("interval_steps", 100) == 0:
                first_k_dense_replace = gpc.config.model.moe_kwargs.get("first_k_dense_replace", 0)
                moe_layer_freq = gpc.config.model.moe_kwargs.get("moe_layer_freq", 1)
                if moe_layer_freq == 1:
                    offset = first_k_dense_replace
                else:
                    offset = first_k_dense_replace + moe_layer_freq - first_k_dense_replace % moe_layer_freq
                micro_num = gpc.config.data.micro_num
                num_experts = gpc.config.model.moe_kwargs.get("num_experts")
                if moe_monitor_cfg.get("logit_before_gate", False):
                    logit_before_gate_max = torch.stack(gpc.metric["logit_before_gate_max"], dim=0).view(micro_num, -1)
                    logit_before_gate_min = torch.stack(gpc.metric["logit_before_gate_min"], dim=0).view(micro_num, -1)
                    logit_before_gate_mean = torch.stack(gpc.metric["logit_before_gate_mean"], dim=0).view(
                        micro_num, -1
                    )
                    if gpc.is_initialized(ParallelMode.PIPELINE):
                        logit_stack = torch.stack(
                            (logit_before_gate_max, logit_before_gate_min, logit_before_gate_mean)
                        )
                        output = all_gather_moe_metrics_pp(logit_stack, offset, gather_dim=2)
                        logit_before_gate_max = output[0].contiguous()
                        logit_before_gate_min = output[1].contiguous()
                        logit_before_gate_mean = output[2].contiguous()

                    dist.all_reduce(
                        logit_before_gate_max, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL)
                    )
                    dist.all_reduce(
                        logit_before_gate_min, op=dist.ReduceOp.MIN, group=gpc.get_group(ParallelMode.GLOBAL)
                    )
                    dist.all_reduce(
                        logit_before_gate_mean, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.GLOBAL)
                    )

                if moe_monitor_cfg.get("tokens_above_avg", False):
                    tokens_above_avg_max = torch.stack(gpc.metric["tokens_above_avg_max"]).view(
                        micro_num, -1, num_experts
                    )
                    tokens_above_avg_min = torch.stack(gpc.metric["tokens_above_avg_min"]).view(
                        micro_num, -1, num_experts
                    )
                    if gpc.is_initialized(ParallelMode.PIPELINE):
                        tokens_above_avg_stack = torch.stack((tokens_above_avg_max, tokens_above_avg_min))
                        output = all_gather_moe_metrics_pp(tokens_above_avg_stack, offset, gather_dim=2)
                        tokens_above_avg_max = output[0]
                        tokens_above_avg_min = output[1]
                    # expert id
                    max_expert_indices = tokens_above_avg_max.max(dim=0).values.max(dim=1).indices
                    min_expert_indices = tokens_above_avg_min.min(dim=0).values.min(dim=1).indices

                    tokens_above_avg_max = tokens_above_avg_max.permute(0, 2, 1).reshape(micro_num * num_experts, -1)
                    tokens_above_avg_min = tokens_above_avg_min.permute(0, 2, 1).reshape(micro_num * num_experts, -1)

                if moe_monitor_cfg.get("expert_activation", False):
                    expert_activation = torch.stack(gpc.metric["expert_activation"]).view(micro_num, -1, num_experts)
                    if gpc.is_initialized(ParallelMode.PIPELINE):
                        output = all_gather_moe_metrics_pp(expert_activation, offset, gather_dim=1)
                        expert_activation = output
                    # get per expert total activations: [num_layers, num_experts]
                    expert_activation = expert_activation.sum(dim=0)

    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)

        num_consumed_images = torch.tensor(
            # sum([image_flags.sum().item() for image_flags in batch[0]["image_flags"]]),
            sum(
                [image_flags.sum().item() if image_flags is not None else 0 for image_flags in batch[0]["image_flags"]]
            ) if "image_flags" in batch[0] else 0,
            dtype=torch.long,
            device=get_current_device(),
        )
        num_consumed_gen_images = torch.tensor(
            sum(
                [image_for_gen_loss_flags.sum().item() if image_for_gen_loss_flags is not None else 0 for image_for_gen_loss_flags in batch[0]["image_for_gen_loss_flags"]]
            ) if "image_for_gen_loss_flags" in batch[0] else 0,
            dtype=torch.long,
            device=get_current_device(),
        )
        num_consumed_text_tokens = torch.tensor(
            (batch[1] != LabelSmoother.ignore_index).sum().item(), dtype=torch.long, device=get_current_device()
        )
        num_padding_tokens = torch.tensor(num_padding_tokens, dtype=torch.long, device=get_current_device())

        cu_seqlens = batch[0]['cu_seqlens'][0]
        type_ids = metric.type_ids.flatten()
        num_gen_samples_t2i = 0
        num_gen_samples_editing = 0
        num_gen_samples_interleave = 0
        for sample_start_idx in cu_seqlens[:-1]:
            cur_sample_type_id = type_ids[sample_start_idx]
            if cur_sample_type_id == 3:
                num_gen_samples_t2i += 1
            elif cur_sample_type_id == 4:
                num_gen_samples_editing += 1
            elif cur_sample_type_id == 5:
                num_gen_samples_interleave += 1

        num_gen_samples_t2i = torch.tensor(num_gen_samples_t2i, dtype=torch.long, device=get_current_device())
        num_gen_samples_editing = torch.tensor(num_gen_samples_editing, dtype=torch.long, device=get_current_device())
        num_gen_samples_interleave = torch.tensor(num_gen_samples_interleave, dtype=torch.long, device=get_current_device())

        dist.all_reduce(num_consumed_images, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(num_consumed_gen_images, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(num_consumed_text_tokens, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(num_padding_tokens, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))

        dist.all_reduce(num_gen_samples_t2i, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(num_gen_samples_editing, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(num_gen_samples_interleave, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))

        train_state.num_consumed_images += num_consumed_images.item()
        train_state.num_consumed_gen_images += num_consumed_gen_images.item()
        train_state.num_consumed_text_tokens += num_consumed_text_tokens.item()
        train_state.num_padding_tokens += num_padding_tokens.item()

        train_state.num_gen_samples_t2i += num_gen_samples_t2i.item()
        train_state.num_gen_samples_editing += num_gen_samples_editing.item()
        train_state.num_gen_samples_interleave += num_gen_samples_interleave.item()

        if loss is not None:
            dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))

        if moe_loss is not None:
            dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))

        if moe_z_loss is not None:
            dist.all_reduce(moe_z_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
        if moe_coef_loss is not None:
            dist.all_reduce(moe_coef_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
        if image_gen_loss is not None:
            dist.all_reduce(image_gen_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
        if losses_for_log_only is not None:
            for loss_for_log_only in losses_for_log_only.values():
                dist.all_reduce(loss_for_log_only, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))

        if "consumed_samples" in kwargs:
            if loss is not None:
                consumed_samples_iter = torch.tensor(kwargs["consumed_samples"], device=loss.device)
                dist.all_reduce(consumed_samples_iter, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))
                train_state.consumed_samples += consumed_samples_iter.item()
                kwargs.pop("consumed_samples")

        # moe monitor
        if gpc.config.model.use_moe:
            assert moe_loss is not None, f"moe_loss should not be None when training {gpc.config.model_type}. "

            if batch_count % moe_monitor_cfg.get("interval_steps", 10) == 0:
                first_k_dense_replace = gpc.config.model.moe_kwargs.get("first_k_dense_replace", 0)
                first_layer = gpc.config.model.first_layer
                first_layer = max(first_layer, first_k_dense_replace)
                if gpc.config.model.moe_location == "llm" and gpc.get_local_rank(ParallelMode.DATA) == 0:
                    num_layers = gpc.config.model.last_layer - first_layer
                    if num_layers > 0:
                        for i in range(num_layers):
                            if moe_monitor_cfg.get("layer_moe_loss", False) and len(gpc.metric["moe_loss"]) > i:
                                writer.add_scalar(
                                    key=f"moe_loss/llm_layer{i+first_layer}",
                                    value=gpc.metric["moe_loss"][i].item(),
                                    step=train_state.step_count,
                                )
                            if moe_monitor_cfg.get("layer_z_loss", False) and len(gpc.metric["moe_z_loss"]) > i:
                                writer.add_scalar(
                                    key=f"moe_z_loss/llm_layer{i+first_layer}",
                                    value=gpc.metric["moe_z_loss"][i].item(),
                                    step=train_state.step_count,
                                )
                            if moe_monitor_cfg.get("layer_coef_loss", False) and len(gpc.metric["moe_coef_loss"]) > i:
                                writer.add_scalar(
                                    key=f"moe_coef_loss/llm_layer{i+first_layer}",
                                    value=gpc.metric["moe_coef_loss"][i].item(),
                                    step=train_state.step_count,
                                )
                            if moe_monitor_cfg.get("route_coef", False) and len(gpc.metric["moe_route_coef"]) > i:
                                writer.add_scalar(
                                    key=f"moe_route_coef/llm_layer{i+first_layer}",
                                    value=(
                                        gpc.metric["moe_route_coef"][i]
                                        if isinstance(gpc.metric["moe_route_coef"][i], float)
                                        else gpc.metric["moe_route_coef"][i].item()
                                    ),
                                    step=train_state.step_count,
                                )
                            if moe_monitor_cfg.get("gates_max", False) and len(gpc.metric["gates_max"]) > i:
                                writer.add_scalar(
                                    key=f"moe_gates_max/llm_layer{i+first_layer}",
                                    value=gpc.metric["moe_gates_max"][i].item(),
                                    step=train_state.step_count,
                                )
                            if moe_monitor_cfg.get("drop_ratio", False) and len(gpc.metric["moe_drop_ratio"]) > i:
                                writer.add_scalar(
                                    key=f"moe_drop_ratio/llm_layer{i+first_layer}",
                                    value=gpc.metric["moe_drop_ratio"][i].item(),
                                    step=train_state.step_count,
                                )
                if gpc.config.model.moe_location == "vision" and gpc.get_local_rank(ParallelMode.DATA) == 0:
                    num_layers = gpc.config.model.vit_cfg["num_hidden_layers"]
                    for i in range(num_layers):
                        if moe_monitor_cfg.get("layer_moe_loss", False) and len(gpc.metric["moe_loss"]) > i:
                            writer.add_scalar(
                                key=f"moe_loss/vision_layer{i}",
                                value=gpc.metric["moe_loss"][i].item(),
                                step=train_state.step_count,
                            )
                        if moe_monitor_cfg.get("layer_z_loss", False) and len(gpc.metric["moe_z_loss"]) > i:
                            writer.add_scalar(
                                key=f"moe_z_loss/vision_layer{i}",
                                value=gpc.metric["moe_z_loss"][i].item(),
                                step=train_state.step_count,
                            )
                        if moe_monitor_cfg.get("layer_coef_loss", False) and len(gpc.metric["moe_coef_loss"]) > i:
                            writer.add_scalar(
                                key=f"moe_coef_loss/vision_layer{i}",
                                value=gpc.metric["moe_coef_loss"][i].item(),
                                step=train_state.step_count,
                            )
                        if moe_monitor_cfg.get("route_coef", False) and len(gpc.metric["moe_route_coef"]) > i:
                            writer.add_scalar(
                                key=f"moe_route_coef/vision_layer{i}",
                                value=gpc.metric["moe_route_coef"][i].item(),
                                step=train_state.step_count,
                            )
                        if moe_monitor_cfg.get("gates_max", False) and len(gpc.metric["gates_max"]) > i:
                            writer.add_scalar(
                                key=f"moe_gates_max/vision_layer{i}",
                                value=gpc.metric["moe_gates_max"][i].item(),
                                step=train_state.step_count,
                            )
                        if moe_monitor_cfg.get("drop_ratio", False) and len(gpc.metric["moe_drop_ratio"]) > i:
                            writer.add_scalar(
                                key=f"moe_drop_ratio/vision_layer{i}",
                                value=gpc.metric["moe_drop_ratio"][i].item(),
                                step=train_state.step_count,
                            )
    else:
        print(f"rank {dist.get_rank()} success_update: {success_update} loss {loss}", flush=True)

    if moe_monitor_cfg.get("layer_moe_loss", False):
        gpc.metric["moe_loss"] = []
    if moe_monitor_cfg.get("layer_z_loss", False):
        gpc.metric["moe_z_loss"] = []
    if moe_monitor_cfg.get("layer_coef_loss", False):
        gpc.metric["moe_coef_loss"] = []
    if moe_monitor_cfg.get("route_coef", False):
        gpc.metric["moe_route_coef"] = []
    if moe_monitor_cfg.get("gates_max", False):
        gpc.metric["moe_gates_max"] = []
    if moe_monitor_cfg.get("drop_ratio", False):
        gpc.metric["moe_drop_ratio"] = []

    if gpc.is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if batch_count != 0 and batch_count % writer.queue_max_length == 0:
        async_hook(manual_submit_count=8)

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(optimizer, "grad_scaler"):
            scaler = optimizer.grad_scaler._scale.item()
        elif hasattr(optimizer.optim, "grad_scaler"):
            scaler = optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        time_cost = time.time() - start_time
        tk_per_gpu = round(
            num_tokens_in_batch * gpc.get_world_size(ParallelMode.DATA) / gpc.get_world_size(ParallelMode.GLOBAL),
            4,
        )
        tgs_statistic = train_state.tgs_statistic
        tgs_statistic["sum_step"] += 1
        tgs_statistic["sum_tg"] += tk_per_gpu
        tgs_statistic["total_time"] = time.time() - very_begining_time
        tgs_statistic["sum_last_tg_10"] += tk_per_gpu
        tgs_statistic["sum_last_time_10"] += time_cost
        tgs_statistic["sum_last_tg_50"] += tk_per_gpu
        tgs_statistic["sum_last_time_50"] += time_cost
        tgs_statistic["SMA_tg_50"] += tk_per_gpu
        tgs_statistic["SMA_time_50"] += time_cost
        tgs_statistic["SMA_tg_50_list"].append(tk_per_gpu)
        tgs_statistic["SMA_time_50_list"].append(time_cost)
        if tgs_statistic["sum_step"] > 50:
            tgs_statistic["SMA_tg_50"] -= tgs_statistic["SMA_tg_50_list"][0]
            tgs_statistic["SMA_time_50"] -= tgs_statistic["SMA_time_50_list"][0]
            tgs_statistic["SMA_tg_50_list"].popleft()
            tgs_statistic["SMA_time_50_list"].popleft()

        last_tgs_1 = round(tk_per_gpu / time_cost, 2)
        tgs_statistic["sum_tgs"] += last_tgs_1

        if tgs_statistic["sum_step"] % 10 == 0:
            tgs_statistic["last_tgs_10"] = round(tgs_statistic["sum_last_tg_10"] / tgs_statistic["sum_last_time_10"], 2)
            tgs_statistic["sum_last_tg_10"] = 0
            tgs_statistic["sum_last_time_10"] = 0

        if tgs_statistic["sum_step"] % 50 == 0:
            tgs_statistic["last_tgs_50"] = round(tgs_statistic["sum_last_tg_50"] / tgs_statistic["sum_last_time_50"], 2)
            tgs_statistic["sum_last_tg_50"] = 0
            tgs_statistic["sum_last_time_50"] = 0

        last_tgs_10 = tgs_statistic["last_tgs_10"]
        last_tgs_50 = tgs_statistic["last_tgs_50"]

        tgs_all = round(tgs_statistic["sum_tg"] / tgs_statistic["total_time"], 2)
        tgs_avg = round(tgs_statistic["sum_tgs"] / tgs_statistic["sum_step"], 2)
        tgs_SMA = round(tgs_statistic["SMA_tg_50"] / tgs_statistic["SMA_time_50"], 2)

        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / (time.time() - start_time),
            2,
        )

        tgs_origin_wo_padding = round(
            (num_tokens_in_batch - num_padding_tokens_local)
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / (time.time() - start_time),
            2,
        )


        # process grad norm

        if isinstance(grad_norm, dict):
            grad_norm_dict = {f"grad_norm/{k}": v for k, v in grad_norm.items()}
        else:
            grad_norm_dict = {"grad_norm": grad_norm}

        infos = {
            "step": batch_count,
            "loss": loss.item(),
            "loss_reduce": loss_avg.item() if loss_avg > 0.0 else 0.0,
            "tgs (tokens/gpu/second)": tgs_origin,
            "tgs_wo_padding (tokens/gpu/second)": tgs_origin_wo_padding,
            "tgs/last_tgs_1": last_tgs_1,
            "tgs/tgs_all": tgs_all,
            "tgs/tgs_avg": tgs_avg,
            "tgs/tgs_SMA": tgs_SMA,
            "tgs/last_tgs_10": last_tgs_10,
            "tgs/last_tgs_50": last_tgs_50,
            "lr": lr,
            "loss_scale": scaler,
            **grad_norm_dict,
            **kwargs,
        }

        if os.getenv("CI_CD", "False") == "True":
            global ci_loss_list  # pylint: disable=W0602
            ci_loss_list.append(loss_avg.item() if loss_avg > 0.0 else 0.0)

        if mtp_loss is not None and gpc.config.loss.get("mtp_loss_coeff", 0.0) > 0.0:
            infos["mtp_loss"] = mtp_loss.item() / gpc.config.loss.mtp_loss_coeff

        if moe_loss is not None and gpc.config.loss.get("expert_balance_coef", 0.0) != 0.0:
            infos["moe_loss"] = moe_loss.item() / gpc.config.loss.expert_balance_coef
        if moe_z_loss is not None and gpc.config.loss.get("moe_z_loss_coef", 0.0) > 0.0:
            infos["moe_z_loss"] = moe_z_loss.item() / gpc.config.loss.moe_z_loss_coef

        if moe_coef_loss is not None and gpc.config.loss.get("expert_coef_balance_coef", 0.0) > 0.0:
            infos["moe_coef_loss"] = moe_coef_loss.item() / gpc.config.loss.expert_coef_balance_coef
        if image_gen_loss is not None and gpc.config.loss.get("image_gen_loss_coef", 0.0) != 0.0:
            infos["image_gen_loss"] = image_gen_loss.item() / gpc.config.loss.image_gen_loss_coef
        if losses_for_log_only is not None and gpc.config.loss.get("image_gen_loss_coef", 0.0) != 0.0:
            for loss_for_log_only_name, loss_for_log_only_value in losses_for_log_only.items():
                infos[loss_for_log_only_name] = loss_for_log_only_value.item() / gpc.config.loss.image_gen_loss_coef
            

        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["num_consumed_images"] = train_state.num_consumed_images
        infos["num_consumed_gen_images"] = train_state.num_consumed_gen_images
        infos["num_gen_samples_t2i"] = train_state.num_gen_samples_t2i
        infos["num_gen_samples_editing"] = train_state.num_gen_samples_editing
        infos["num_gen_samples_interleave"] = train_state.num_gen_samples_interleave
        infos["consumed_samples"] = train_state.consumed_samples
        infos["num_consumed_text_tokens"] = train_state.num_consumed_text_tokens
        infos["num_padding_tokens"] = train_state.num_padding_tokens
        infos["num_padding_tokens_local"] = num_padding_tokens_local
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        # moe monitor
        if gpc.config.model.use_moe:
            if batch_count % moe_monitor_cfg.get("interval_steps", 100) == 0:
                first_k_dense_replace = gpc.config.model.moe_kwargs.get("first_k_dense_replace", 0)
                moe_layer_freq = gpc.config.model.moe_kwargs.get("moe_layer_freq", 1)
                if moe_layer_freq == 1:
                    offset = first_k_dense_replace
                else:
                    offset = first_k_dense_replace + moe_layer_freq - first_k_dense_replace % moe_layer_freq
                if moe_monitor_cfg.get("tokens_above_avg", False):
                    tokens_above_avg_max = tokens_above_avg_max.max(dim=0).values.to("cpu").tolist()
                    tokens_above_avg_min = tokens_above_avg_min.min(dim=0).values.to("cpu").tolist()
                    max_expert_indices = max_expert_indices.to("cpu").tolist()
                    min_expert_indices = min_expert_indices.to("cpu").tolist()

                    for i in range(len(tokens_above_avg_max)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"tokens_above_avg_max/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=tokens_above_avg_max[i], step=train_state.step_count)
                    for i in range(len(tokens_above_avg_min)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"tokens_above_avg_min/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=tokens_above_avg_min[i], step=train_state.step_count)
                    for i in range(len(max_expert_indices)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"expert_with_max_tokens/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=max_expert_indices[i], step=train_state.step_count)
                    for i in range(len(min_expert_indices)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"expert_with_min_tokens/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=min_expert_indices[i], step=train_state.step_count)

                if moe_monitor_cfg.get("logit_before_gate", False):
                    logit_before_gate_min = logit_before_gate_min.min(dim=0).values.to("cpu").tolist()
                    logit_before_gate_max = logit_before_gate_max.max(dim=0).values.to("cpu").tolist()
                    logit_before_gate_mean = logit_before_gate_mean.mean(dim=0).to("cpu").tolist()
                    for i in range(len(logit_before_gate_min)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"logit_before_gate_min/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=logit_before_gate_min[i], step=train_state.step_count)

                    for i in range(len(logit_before_gate_max)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"logit_before_gate_max/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=logit_before_gate_max[i], step=train_state.step_count)

                    for i in range(len(logit_before_gate_mean)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"logit_before_gate_mean/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=logit_before_gate_mean[i], step=train_state.step_count)

                if moe_monitor_cfg.get("expert_activation", False):
                    expert_activation_min = expert_activation.min(dim=1).values.to("cpu").tolist()
                    expert_activation_max = expert_activation.max(dim=1).values.to("cpu").tolist()
                    expert_activation_mean = expert_activation.float().mean(dim=1).ceil().to("cpu").tolist()

                    for i in range(len(expert_activation_min)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"expert_activation_min/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=expert_activation_min[i], step=train_state.step_count)

                    for i in range(len(expert_activation_max)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"expert_activation_max/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=expert_activation_max[i], step=train_state.step_count)

                    for i in range(len(expert_activation_mean)):
                        layer_idx = i * moe_layer_freq + offset
                        key = f"expert_activation_mean/layer_{layer_idx}"
                        writer.add_scalar(key=key, value=expert_activation_mean[i], step=train_state.step_count)

        writer_interval_step = gpc.config.monitor.tensorboard.get("interval_step", 1)
        if batch_count % writer_interval_step == 0:
            for key, value in infos.items():
                if isinstance(value, dict):
                    writer.add_scalars(key=key, value=value, step=train_state.step_count)
                else:
                    writer.add_scalar(key=key, value=value, step=train_state.step_count)

        line = ""
        for key, value in infos.items():
            if key not in acc_perplex:
                line += f"{key}={value} "

        logger.info(line)

    if gpc.config.model.use_moe:
        if moe_monitor_cfg.get("tokens_above_avg", False):
            gpc.metric["tokens_above_avg_max"] = []
            gpc.metric["tokens_above_avg_min"] = []
        if moe_monitor_cfg.get("logit_before_gate", False):
            gpc.metric["logit_before_gate_max"] = []
            gpc.metric["logit_before_gate_min"] = []
            gpc.metric["logit_before_gate_mean"] = []
        if moe_monitor_cfg.get("expert_activation", False):
            gpc.metric["expert_activation"] = []


def record_execution_times(logger):
    fields = {}
    device = get_current_device()

    for timer_name, timer_value in etc.timers.items():
        timer_value = torch.tensor(timer_value).to(device)
        dist.all_reduce(timer_value, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL))
        fields[timer_name] = timer_value.item()

    if gpc.is_rank_for_log():
        logger.info(fields)
