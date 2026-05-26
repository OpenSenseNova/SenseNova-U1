# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import gc
import logging
import os
import time

from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import get_scheduler

from sensenovalm.checkpoint.checkpoint_manager import CheckpointManager
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.context.process_group_initializer import ParallelMode
from sensenovalm.core.model_average import AveragedModel
from sensenovalm.core.trainer import Trainer
from sensenovalm.data.train_state import get_train_state
from sensenovalm.initialize.initialize_trainer import initialize_trainer
from sensenovalm.model.losses.ce_loss import FlashGPTLMLoss
from sensenovalm.model.metrics import AccPerplex
from sensenovalm.train.pipeline import (  # record_current_batch_training_metrics,
    get_scheduler_hooks,
    initialize_llm_profile,
    initialize_optimizer,
    initialize_parallel_communicator,
    inject_model,
    load_new_batch_with_train_state,
)
from sensenovalm.utils.common import (  # get_megatron_flops,
    BatchSkipper,
    check_cuda_env,
    enable_pytorch_expandable_segments,
    get_current_device,
    launch_time,
)
from sensenovalm.utils.gputest import empty_cache_and_diag
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.megatron_timers import megatron_timer as timer
from sensenovalm.utils.parallel import (
    check_parallel_statistic_equality,
    get_parallel_log_file_name,
)
from sensenovalm.utils.writer import Writer
from sensenovavl.train.record_metrics import record_current_batch_training_metrics

# global llm logger
logger = logging.getLogger(__file__)

_DATALOADER_WORKER_FAILURE_HINTS = (
    "DataLoader worker",
    "dataloader's workers are out of shared memory",
    "Bus error",
    "exited unexpectedly",
    "killed by signal",
    "Connection reset by peer",
)


def _is_dataloader_worker_failure(exc: BaseException) -> bool:
    message = str(exc)
    return any(hint in message for hint in _DATALOADER_WORKER_FAILURE_HINTS)


def sensenovavl_resume_dataloader(train_dl, train_state, ckpt_manager, dataloader_resume_mode):
    dl_loader_custom_infos = {}

    if (
        dataloader_resume_mode == "v1"
        and gpc.config.data.type
        in [
            "multimodal_packed_streaming",
            "multimodal_streaming",
        ]
        and gpc.config.ckpt.get("resume_ds", True)
        and ckpt_manager.load_ckpt_info["path"] is not None
        and gpc.get_local_rank(ParallelMode.COMMON_DATA) == 0
    ):
        logger.info("resume dataloader")

        ds_state = {}
        path = ckpt_manager.load_ckpt_info["path"]
        # NOTE:
        if gpc.config.ckpt.get("auto_resume", True):
            assert path.startswith("local:")
            path = path[len("local:") :]
        
        if gpc.is_rank_for_log():
            logger.info(f"rank [{dist.get_rank()}] resume ds state from :{path}...")

        ds_state_file = os.path.join(path, "ds_state", f"ds_state_{dist.get_rank()}.pt")
        local_ds_state = torch.load(ds_state_file, map_location="cpu")

        if os.path.exists(os.path.join(path, "ds_state", f"ds_info_{dist.get_rank()}.pt")):
            local_info = torch.load(os.path.join(path, "ds_state", f"ds_info_{dist.get_rank()}.pt"))
            dl_loader_custom_infos.update(local_info)

        for ds_name, worker_states in local_ds_state.items():
            if ds_name not in ds_state:
                ds_state[ds_name] = {}

            for worker_key, worker_state in worker_states.items():
                assert worker_key not in ds_state[ds_name]
                ds_state[ds_name][worker_key] = {}
                ds_state[ds_name][worker_key].update(worker_state)

        # NOTE:
        if not gpc.config.ckpt.get("auto_resume", False) and 'llm_mixed_training' in ds_state:
            tmp_ds_state = {}
            tmp_ds_state['llm_mixed_training'] = ds_state['llm_mixed_training'] 
            train_state.ds_state = tmp_ds_state
            logger.info('Only resuming the llm_mixed_training data')
        else:
            train_state.ds_state = ds_state
        
        train_dl.dataset.load_state_dict(train_state.ds_state, custom_infos=dl_loader_custom_infos)

    return dl_loader_custom_infos


def sensenovavl_process_batch_data(batch, train_dl, train_state, dl_loader_custom_infos):
    # record state dict of ds to enable quick resume
    if "worker_state_key_list" in batch[0]:
        ignore_key_list = [
            "sample_info",
        ]
        worker_state_key_list = batch[0].pop("worker_state_key_list")
        worker_state_dict_list = batch[0].pop("worker_state_dict_list")
        worker_state_custom_infos_list = batch[0].pop("worker_state_custom_infos_list")



        for worker_state_key, worker_state_dict, worker_state_custom_infos in zip(
            worker_state_key_list, worker_state_dict_list, worker_state_custom_infos_list
        ):
            # Deserialize if worker_state_dict was pickle-serialized in postprocess_buffer
            if isinstance(worker_state_dict, bytes):
                import pickle
                worker_state_dict = pickle.loads(worker_state_dict)
            for ds_name in worker_state_dict:
                if ds_name in ignore_key_list:
                    continue

                if ds_name not in train_state.ds_state:
                    train_state.ds_state[ds_name] = {}

                if worker_state_key in worker_state_dict[ds_name]:
                    train_state.ds_state[ds_name][worker_state_key] = {}
                    train_state.ds_state[ds_name][worker_state_key].update(worker_state_dict[ds_name][worker_state_key])
            if worker_state_custom_infos is not None:
                # Deserialize if custom_infos values were pickle-serialized
                deserialized = {}
                for ci_key, ci_val in worker_state_custom_infos.items():
                    if isinstance(ci_val, bytes):
                        import pickle
                        deserialized[ci_key] = pickle.loads(ci_val)
                    else:
                        deserialized[ci_key] = ci_val
                dl_loader_custom_infos.update(deserialized)

    # sync data
    if gpc.get_world_size(ParallelMode.COMMON_DATA) > 1:
        dist.broadcast_object_list(
            batch,
            src=gpc.get_ranks_in_group(ParallelMode.COMMON_DATA)[0],
            group=gpc.get_group(ParallelMode.COMMON_DATA),
        )
        if os.environ.get("DEBUG_TYPE", "null") == "pipe_debug":
            logger.info(
                f"step:{train_state.step_count} global rank [{dist.get_rank()}] "
                f"model rank:{gpc.get_local_rank(ParallelMode.COMMON_DATA)} ",
                f"info: {batch[0]['image_flags']}",
            )

    # check batch shape
    for key, value in batch[0].items():
        if isinstance(value, (int, dict, str)):
            continue
        assert (
            len(value) == gpc.config.data.micro_num
        ), f"len of data {key}:{len(value)} != config.micro_num:{gpc.config.data.micro_num}"
    assert (isinstance(batch[1], dict) and len(batch[1]["labels"]) == gpc.config.data.micro_num) or (
        not isinstance(batch[1], dict) and len(batch[1]) == gpc.config.data.micro_num
    ), f"len of label {len(batch[1])} != config.micro_num:{gpc.config.data.micro_num}"

    # we pad dataset to ensure each sample can be seen only one time per epoch
    # here we check if dataset of this rank is exhausted and yield empty data now
    check_empty_data = getattr(train_dl.dataset, "allow_empty_data", False)

    is_empty_data = batch[0].pop("is_empty_data_list", False)
    if is_empty_data:
        assert check_empty_data
        batch[0].pop("worker_state_key_list", None)
        batch[0].pop("worker_state_dict_list", None)


class TrainerBuilder(Trainer):
    """
    A class for building and managing InternEvo training workflow.

    `TrainerBuilder` extends the base `Trainer` class to include additional functionality
    for initializing and managing various components involved in the training process.
    This includes setting up logging, checkpoints, loss functions, optimizers, metrics,
    train states, and profiling tools. The class supports distributed training and allows
    for seamless management of training, evaluation, and checkpointing.

    Args:
        model (Union[torch.nn.Module, List[torch.nn.Module]]): The model to be trained.
        train_dl (DataLoader): DataLoader for training data.
        val_dls (Optional[Dict[str, DataLoader]], optional): DataLoaders for validation data.
        **kwargs: Additional keyword arguments including:
            - config (str): Path to the configuration file.
            - profiling (bool): Whether to enable profiling.
            - dataset_types (list): List of dataset types to be used for training.

    Methods:
        __init__: Initializes the `TrainerBuilder` with the model, data loaders, and other components.
        fit: Runs the training loop, processing batches and handling evaluation and checkpointing.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        train_dl: DataLoader,
        val_dls: Optional[Dict[str, DataLoader]] = None,
        **kwargs,
    ):
        """
        Initialize TrainerBuilder with necessary components for training.

        Args:
            model (Union[torch.nn.Module, List[torch.nn.Module]]): The model to be trained.
            train_dl (DataLoader): DataLoader for training data.
            val_dls (Optional[Dict[str, DataLoader]], optional): DataLoaders for validation data.
            **kwargs: Additional keyword arguments including:
                - config (str): Path to the configuration file.
                - profiling (bool): Whether to enable profiling.
                - dataset_types (list): List of dataset types to be used for training.

        """
        # set very_beginning_time
        self.very_beginning_time = time.time()
        # sensenovavl updates the very beginning time
        if "very_beginning_time" in kwargs:
            self.very_beginning_time = kwargs["very_beginning_time"]
        # broadcast current_time and setup logging
        self.current_time = self._setup_time_and_logging()
        # load config_lines
        config_lines = self._read_config(kwargs["config"])

        # inject model for amp and parallel training
        model = inject_model(model)

        # set torch expandable_segments
        enable_pytorch_expandable_segments()

        # initialize loss function
        criterion = self._initialize_criterion()

        # initialize mtp loss function
        mtp_criterions = self._initialize_mtp_criterion()

        # initialize isp communicator
        isp_communicator = initialize_parallel_communicator(model)

        # initialize train state
        train_state = get_train_state(train_dl)

        # initialize optimizer
        optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

        # sensenovavl customized lr_scheduler
        lr_scheduler_cfg = gpc.config.lr_scheduler
        warmup_steps = lr_scheduler_cfg.init_steps + lr_scheduler_cfg.warmup_ratio * lr_scheduler_cfg.total_steps

        lr_scheduler_offset = int(gpc.config.get('lr_scheduler_offset', 0))

        # NOTE:
        if gpc.config.lr_scheduler_type == 'cosine_with_min_lr':
            lr_scheduler = get_scheduler(gpc.config.lr_scheduler_type, 
                                         optimizer, warmup_steps, 
                                         lr_scheduler_cfg.total_steps-lr_scheduler_offset, 
                                         {'min_lr_rate': gpc.config.min_lr_ratio})
        else:
            lr_scheduler = get_scheduler(gpc.config.lr_scheduler_type, 
                                         optimizer, warmup_steps, 
                                         lr_scheduler_cfg.total_steps-lr_scheduler_offset)

        print(f'using lr scheduler {gpc.config.lr_scheduler_type}')

        # initialize checkpoint manager and try resume training
        # NOTE: create averaged model AFTER resume so it tracks the correct (injected) model weights.
        self.ckpt_manager = self._initialize_checkpoint_manager(model, optimizer, lr_scheduler, train_dl, config_lines)
        self.ckpt_manager.try_resume_training(train_state, self.current_time)
        check_parallel_statistic_equality(model)

        averaged_model = None
        if "averaged_model" in gpc.config and gpc.config.averaged_model.get("enable", False):
            use_buffers = gpc.config.averaged_model.get("use_buffers", False)
            decay = gpc.config.averaged_model.get("decay", 0.999)
            multi_avg_fn_str = gpc.config.averaged_model.get("multi_avg_fn", None)
            if multi_avg_fn_str == "ema":
                multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(decay)
            elif multi_avg_fn_str == "swa":
                multi_avg_fn = torch.optim.swa_utils.get_swa_multi_avg_fn(decay)
            else:
                raise ValueError(f"Invalid multi_avg_fn: {multi_avg_fn_str}, only support ema and swa")

            averaged_model = AveragedModel(
                model.model,
                multi_avg_fn=multi_avg_fn,
                use_buffers=use_buffers,
            )
            # Bind and resume averaged model weights from `<ckpt>/averaged_model/` if present.
            self.ckpt_manager.set_averaged_model(averaged_model)
            self.ckpt_manager.try_load_averaged_model_from_checkpoint()

        # sensenovavl loading other persistent training states
        assert "dataloader_resume_mode" in kwargs, "dataloader_resume_mode should be set."
        self.sensenovavl_dl_resume_mode = kwargs["dataloader_resume_mode"]
        self.sensenovavl_dl_custom_infos = sensenovavl_resume_dataloader(
            train_dl, train_state, self.ckpt_manager, self.sensenovavl_dl_resume_mode
        )
        # sensenovavl records consumed samples num
        self.sensenovavl_consumed_samples = 0

        # initialize customed llm writer
        self.writer = self._initialize_writer(train_state, config_lines)

        # initialize metric for calculating accuracy and perplexity
        self.metric = self._initialize_metric(kwargs["dataset_types"])

        # initialize batch skipper
        self.batch_skipper = self._initialize_batch_skipper(train_state)

        # initialize trainer
        engine, scheduler = initialize_trainer(
            model=model,
            averaged_model=averaged_model,
            optimizer=optimizer,
            criterion=criterion,
            mtp_criterions=mtp_criterions,
            lr_scheduler=lr_scheduler,
            beta2_scheduler=beta2_scheduler,
            scheduler_hooks=get_scheduler_hooks(self.metric, optimizer, isp_communicator),
        )

        # set attributes
        self._set_attributes(
            kwargs["profiling"], train_dl, val_dls, train_state, model, optimizer, beta2_scheduler, isp_communicator
        )

        super().__init__(engine, scheduler)

    def _setup_time_and_logging(self) -> str:
        current_time = launch_time()
        objs = [current_time]
        dist.broadcast_object_list(objs, src=0)
        current_time = objs[0].replace(":", ".")
        global logger
        logger = get_logger(
            __name__, launch_time=current_time, job_name=gpc.config.JOB_NAME, file_name=get_parallel_log_file_name()
        )
        return current_time

    def _read_config(self, config_path: str) -> list:
        with open(config_path, "r") as f:
            return f.readlines()

    def _initialize_criterion(self, ) -> FlashGPTLMLoss:
        return FlashGPTLMLoss(
            parallel_output=gpc.config.model.parallel_output, label_smoothing=gpc.config.loss.label_smoothing, ce_loss_weight=float(gpc.config.get('ce_loss_weight', 1.0))
        )

    def _initialize_mtp_criterion(self) -> FlashGPTLMLoss:
        if hasattr(gpc.config.model, "num_mtp_layers") and gpc.config.model.num_mtp_layers > 0:
            mtp_criterions = []
            for _ in range(gpc.config.model.num_mtp_layers):
                mtp_criterion = FlashGPTLMLoss(
                    parallel_output=gpc.config.model.parallel_output, label_smoothing=gpc.config.loss.label_smoothing
                )
                mtp_criterions.append(mtp_criterion)
        else:
            mtp_criterions = []
        return mtp_criterions

    def _initialize_checkpoint_manager(
        self, model, optimizer, lr_scheduler, train_dl, config_lines
    ) -> CheckpointManager:
        return CheckpointManager(
            ckpt_config=gpc.config.ckpt,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dl=train_dl,
            model_config=gpc.config.model,
            model_config_file="".join(config_lines),
            feishu_address=gpc.config.monitor.alert.feishu_alert_address,
        )

    def _initialize_writer(self, train_state, config_lines) -> Writer:
        return Writer(
            job_name=gpc.config.JOB_NAME,
            launch_time=self.current_time,
            file_name=get_parallel_log_file_name(),
            tensorboard_folder=gpc.config.tensorboard_folder,
            resume_tb_folder=train_state.resume_tb_folder,
            step_count=train_state.step_count,
            config=config_lines,
            logger=logger,
            enable_tb=gpc.config.enable_tb,
            queue_max_length=gpc.config.tensorboard.queue_max_length,
            total_steps=gpc.config.data.total_steps,
            enable_wandb=os.environ.get("WANDB_API_KEY", None) is not None
        )

    def _initialize_metric(self, dataset_types) -> AccPerplex:
        # initialize metric for calculating accuracy and perplexity
        # if isp mode, head output is parallel in sequence dim, metric dp group should be SP*DP
        # _dp_pg = (
        #     gpc.get_group(ParallelMode.ISP_DATA)
        #     if is_using_isp() and gpc.config.model.parallel_output
        #     else gpc.get_group(ParallelMode.DATA)
        # )
        # _tp_pg = dist.new_group([gpc.get_global_rank()]) if is_using_isp() else gpc.get_group(ParallelMode.TENSOR)
        _dp_pg = gpc.get_group(ParallelMode.DATA)
        _tp_pg = gpc.get_group(ParallelMode.TENSOR)
        return AccPerplex(
            device=get_current_device(),
            tp_pg=_tp_pg,
            dp_pg=_dp_pg,
            dataset_types=dataset_types,
        )

    def _initialize_batch_skipper(self, train_state) -> BatchSkipper:
        skip_batches = gpc.config.data.skip_batches
        return BatchSkipper(skip_batches)

    def _set_attributes(
        self, profiling, train_dl, val_dls, train_state, model, optimizer, beta2_scheduler, isp_communicator
    ):
        self.profiling = profiling
        self.train_dl = train_dl
        self.val_dls = val_dls
        self.train_state = train_state
        self.model = model
        self.optimizer = optimizer
        self.beta2_scheduler = beta2_scheduler
        self.isp_communicator = isp_communicator

    def fit(self):
        """
        Run InternEvo training loop.
        """
        self.train()
        train_iter = iter(self.train_dl)
        self.train_start_time = time.perf_counter()

        # sensenovavl update the start step id
        if self.sensenovavl_dl_resume_mode == "v0":
            start_step = 0
        elif self.sensenovavl_dl_resume_mode == "v1":
            start_step = self.train_state.batch_count

        with initialize_llm_profile(profiling=self.profiling, start_time=self.current_time) as prof:
            gc.disable()
            consecutive_worker_failures = 0
            for batch_count in range(start_step, gpc.config.data.total_steps):
                try:
                    should_stop, train_iter = self._process_batch(batch_count, train_iter, prof)
                    consecutive_worker_failures = 0
                except Exception as exc:
                    train_iter, consecutive_worker_failures = self._maybe_recover_dataloader_worker_failure(
                        batch_count=batch_count,
                        train_iter=train_iter,
                        exc=exc,
                        consecutive_worker_failures=consecutive_worker_failures,
                    )
                    continue

                if should_stop:
                    break

        self.ckpt_manager.wait_async_upload_finish()

    def _maybe_recover_dataloader_worker_failure(self, batch_count: int, train_iter, exc, consecutive_worker_failures):
        if not _is_dataloader_worker_failure(exc):
            raise exc

        restart_workers = getattr(self.train_dl, "restart_workers", None)
        if not callable(restart_workers):
            raise exc

        max_retries = max(int(getattr(self.train_dl, "worker_fail_retry", 0)), 0)
        if consecutive_worker_failures >= max_retries:
            logger.error(
                "Dataloader worker failure at batch_count=%s exceeded retry budget (%s): %s",
                batch_count,
                max_retries,
                exc,
            )
            raise exc

        consecutive_worker_failures += 1
        backoff_s = max(float(getattr(self.train_dl, "worker_fail_retry_backoff_s", 0.0)), 0.0)

        logger.warning(
            "Detected dataloader worker failure at batch_count=%s. Restarting worker pool "
            "(attempt %s/%s, backoff=%ss): %s",
            batch_count,
            consecutive_worker_failures,
            max_retries,
            backoff_s,
            exc,
        )

        timer.reset()
        self.zero_grad()
        gc.collect()
        if backoff_s > 0:
            time.sleep(backoff_s)

        train_iter = restart_workers()
        return train_iter, consecutive_worker_failures

    def _process_batch(self, batch_count: int, train_iter, prof):
        empty_cache_and_diag(
            batch_count,
            bcast_sumbit_hook=self.optimizer.submit_bcast_async,
            interval=gpc.config.data.empty_cache_and_diag_interval,
        )
        check_cuda_env()
        start_time = time.time()
        step_start_time = time.perf_counter()
        timer("one-batch").start()

        gpc.config.batch_count = batch_count
        batch, train_iter, num_samples, num_padding_tokens = self._load_and_prepare_batch(batch_count, train_iter)
        if self.batch_skipper(batch_count):
            if gpc.is_rank_for_log():
                logger.info(f"Skip batch count:`{batch_count}`...")
            timer("one-batch").stop()
            return False, train_iter

        timer("fwd-bwd").start()
        loss, mtp_loss, boi_loss, moe_loss, moe_z_loss, moe_coef_loss, image_gen_loss, losses_for_log_only = self._forward_backward(batch)
        timer("fwd-bwd").stop()

        success_update, grad_norm_groups = self._update_parameters()

        step_end_time = time.perf_counter()
        time_per_sample = step_end_time - step_start_time
        time_per_sample_avg = (step_end_time - self.train_start_time) / self.sensenovavl_consumed_samples
        consumed_samples_for_eta = self.sensenovavl_consumed_samples * gpc.get_world_size(ParallelMode.DATA)
        while consumed_samples_for_eta >= 5e6:
            consumed_samples_for_eta -= 5e6
        eta_next_5m = (
            (5e6 - consumed_samples_for_eta) * time_per_sample_avg / 60 / 60 / gpc.get_world_size(ParallelMode.DATA)
        )
        timer("record_metrics").start()
        record_current_batch_training_metrics(
            logger=logger,
            writer=self.writer,
            success_update=success_update,
            batch_count=batch_count,
            batch=batch,
            train_state=self.train_state,
            optimizer=self.optimizer,
            beta2_scheduler=self.beta2_scheduler,
            start_time=start_time,
            very_begining_time=self.very_beginning_time,
            loss=loss,
            mtp_loss=mtp_loss,
            moe_loss=moe_loss,
            moe_z_loss=moe_z_loss,
            moe_coef_loss=moe_coef_loss,
            boi_loss=boi_loss,
            image_gen_loss=image_gen_loss,
            losses_for_log_only=losses_for_log_only,
            grad_norm=grad_norm_groups,
            metric=self.metric,
            num_padding_tokens=num_padding_tokens,
            async_hook=self.optimizer.submit_bcast_async,
            consumed_samples=num_samples,
            time_per_sample=time_per_sample,
            time_per_sample_avg=time_per_sample_avg,
            eta_next_5m=eta_next_5m,
        )
        timer("record_metrics").stop()

        timer("one-batch").stop()
        timer_metrics = [
            "one-batch",
            "batch-gen",
            "fwd",
            "bwd",
            "fwd-bwd",
            "cal_loss",
            "sync_grad",
            "cal_loss",
            "step",
            "record_metrics",
        ]
        if gpc.is_rank_for_log():
            timer.log(timer_metrics, logger, rank=gpc.get_global_rank())

        if self.ckpt_manager.try_save_checkpoint(self.train_state, self.sensenovavl_dl_custom_infos):
            return True, train_iter

        if batch_count % getattr(gpc.config, "check_model_weight_per_steps", 500) == 0:
            check_parallel_statistic_equality(self.model)

        self._update_profilers(batch_count, prof)
        return False, train_iter

    def _load_and_prepare_batch(self, batch_count: int, train_iter):
        batch, train_iter = load_new_batch_with_train_state(train_dl=self.train_dl, train_iter=train_iter, train_state=self.train_state)
        sensenovavl_process_batch_data(batch, self.train_dl, self.train_state, self.sensenovavl_dl_custom_infos)

        num_samples = batch[0].pop("num_samples", 0)
        if num_samples == 0:
            num_samples = (
                sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
                if "cu_seqlens" in batch[0]
                else batch[0]["input_ids"].shape[0]
            )
        num_padding_tokens = batch[0].pop("num_padding_tokens", 0)
        self.sensenovavl_consumed_samples += num_samples

        self.train_state.batch_count = batch_count
        self.train_state.num_consumed_samples_in_epoch += len(batch[1])
        if batch[0].get("type_ids", None) is not None:
            self.metric.set_current_type_ids(type_ids=batch[0].get("type_ids", None))
        return batch, train_iter, num_samples, num_padding_tokens

    def _forward_backward(self, batch):
        self.zero_grad()

        # sensenovavl moe_loss and boi_loss
        moe_loss, moe_z_loss, moe_coef_loss = None, None, None
        boi_loss = None
        mtp_loss = None
        image_gen_loss = None

        outputs = self.execute_schedule(
            batch,
            forward_only=False,
            return_loss=True,
            return_output_label=False,
        )
        if isinstance(outputs[-1], dict):
            losses_for_log_only = outputs[-1]
            _, _, loss, *extra_loss = outputs[:-1]
        else:
            losses_for_log_only = None
            _, _, loss, *extra_loss = outputs[:-1]

        if extra_loss:
            if gpc.config.loss.get("image_gen_loss_coef", 0.0) > 0:
                image_gen_loss = extra_loss.pop(-1)
            if hasattr(gpc.config.model, "num_mtp_layers") and gpc.config.model.num_mtp_layers > 0:
                mtp_loss = extra_loss.pop(0)
            if gpc.config.model.use_moe:
                moe_loss = extra_loss.pop(0)
                if len(extra_loss) == 1:
                    moe_z_loss = extra_loss[0]
                if len(extra_loss) == 2:
                    moe_z_loss = extra_loss[0]
                    moe_coef_loss = extra_loss[1]

        return loss, mtp_loss, boi_loss, moe_loss, moe_z_loss, moe_coef_loss, image_gen_loss, losses_for_log_only

    def _update_parameters(self):
        trainer_result = self.step()
        assert trainer_result is not None
        success_update, grad_norm_groups = trainer_result
        if success_update:
            self.train_state.step_count += 1
        else:
            self.train_state.inf_nan_skip_batches += 1
            if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():
                logger.warning(f"Warning: skip parameter update at step {self.train_state.batch_count}.")
        return success_update, grad_norm_groups

    def _update_profilers(self, batch_count: int, prof):
        if batch_count % 2 == 0:
            prof.step()
