# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
import os
import socket
import time
from enum import Enum
from functools import partial
from typing import Callable, Dict

import torch

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer import TrainState
from sensenovalm.initialize.launch import get_config_value
from sensenovalm.model.base_model import BaseModel
from sensenovalm.model.registry import model_initializer
from sensenovalm.solver.optimizer import HybridZeroOptimizer, HybridZeroOptimizer_v2
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.megatron_timers import megatron_timer as timer
from sensenovalm.utils.storage_manager import (
    get_storage_manager,
    init_storage_manager,
    llm_load,
    llm_save,
    try_get_storage_backend,
)
from sensenovalm.utils.timeout import llm_timeout

def auto_resume_sanity_check(ckpt_config):
    load_given_ckpt = get_config_value(ckpt_config, "load_given_ckpt", None)
    if load_given_ckpt is None:
        return True  # default value is True
    else:
        return not load_given_ckpt


def ckpt_info_sanity_check(ckpt_config):
    load_ckpt_folder = get_config_value(ckpt_config, "load_ckpt_folder", None)

    load_model_only_folder = get_config_value(ckpt_config, "load_model_only_folder", None)

    if load_model_only_folder is not None:
        assert (
            load_ckpt_folder is None
        ), "Detect 'load_ckpt_folder' and 'load_model_only_folder' set at the same time, \
# and 'load_given_ckpt' is True, so sensenovalm will load from 'load_ckpt_folder'"
        return dict(path=load_model_only_folder, content=("model",), ckpt_type="internevo")
    else:
        load_optimizer = get_config_value(ckpt_config, "load_optimizer", True)

        if isinstance(load_ckpt_folder, str):
            if load_optimizer:
                return dict(path=load_ckpt_folder, content=("model", "sampler", "optimizer"), ckpt_type="internevo")
            else:
                return dict(path=load_ckpt_folder, content=("model", "sampler"), ckpt_type="internevo")
        elif load_ckpt_folder is None:
            return None
        else:
            assert f"Unsupport data type:'{type(load_ckpt_folder)}' for config.ckpt arg: 'load_ckpt_folder'"


from .components import (
    load_context,
    load_model_checkpoint,
    load_model_state_dict,
    load_optimizer_checkpoint,
    load_sampler,
    load_scheduler,
    save_model_checkpoint,
    save_optimizer_checkpoint,
)
from .convert_to_hf import convert_internevo_ckpt_to_hf
from .load_funcs import LOAD_FUNC_DICT
from .utils import process_load_info

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


class CheckpointSaveType(Enum):
    NORMAL_CHECKPOINT = 1
    SNAPSHOT_CHECKPOINT = 2


class CheckpointLoadContent:
    MODEL = "model"
    SAMPLER = "sampler"
    OPIMIZER = "optimizer"
    SCHEDULAER = "scheduler"


def try_load_internevo_ckpt(ckpt_mm, load_info, train_state: TrainState = None):
    """Tries to load a checkpoint from the given folder.

    Args:
        ckpt_mm: CheckpointManager object that contains model, optimizer, etc. to load state into
        load_info: Dict containing path and content info on what to load
        train_state: TrainState object to load trainer state into, optional

    This will selectively load parts of the checkpoint based on the load_content mask:
        - MODEL: Model weights
        - OPTIMIZER: Optimizer state
        - SCHEDULER: Learning rate scheduler state
        - SAMPLER: Data sampler state

    For each part loaded, the corresponding object in ckpt_mm or train_state will be updated.

    Returns: String summarizing what parts were loaded.

    The key functionality is selectively choosing which parts of a checkpoint to load via the load_content mask
        and the checkpoint manager ckpt_mm and train state objects
    """
    load_content_str, load_ckpt_folder, load_content = process_load_info(load_info)

    if load_content.need_load(CheckpointLoadContent.MODEL):
        load_model_checkpoint(folder=load_ckpt_folder, model=ckpt_mm.model)
        load_content_str += f"{CheckpointLoadContent.MODEL}, "

    if load_content.not_only_load(CheckpointLoadContent.MODEL):
        # load training states.
        if train_state:
            load_context(load_ckpt_folder, train_state)

        # load optimizer states.
        if load_content.need_load(CheckpointLoadContent.OPIMIZER):
            load_optimizer_checkpoint(load_ckpt_folder, ckpt_mm.optimizer)
            load_content_str += f"{CheckpointLoadContent.OPIMIZER}, "
        else:
            if gpc.is_rank_for_log():
                logger.warning("CheckpointManager has no 'optimizer', skip reload optim checkpoint!")

        # load lr scheduler states.
        if load_content.need_load(CheckpointLoadContent.SCHEDULAER):
            if ckpt_mm.lr_scheduler and train_state:
                load_scheduler(load_ckpt_folder, ckpt_mm.lr_scheduler, ckpt_mm.optimizer, train_state)
                load_content_str += f"{CheckpointLoadContent.SCHEDULAER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning("CheckpointManager has no 'lr_scheduler', skip reload lr_scheduler checkpoint!")

            if not load_content.need_load(CheckpointLoadContent.OPIMIZER):
                if ckpt_mm.lr_scheduler and train_state:
                    gpc.config.only_load_lr = True
                    load_optimizer_checkpoint(load_ckpt_folder, ckpt_mm.optimizer)
                    gpc.config.only_load_lr = False

        # load dataloader sampler states.
        if load_content.need_load(CheckpointLoadContent.SAMPLER) and train_state:
            if hasattr(train_state, "batch_sampler") and not isinstance(
                train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
            ):
                load_sampler(load_ckpt_folder, ckpt_mm.train_dl.batch_sampler)
                # track the actual updates of sampler when using weighted sampling
                train_state.init_batch_sampler(ckpt_mm.train_dl.batch_sampler)
                load_content_str += f"{CheckpointLoadContent.SAMPLER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning("CheckpointManager skip reload 'batch_sampler'")

            # reload data state dict.
            if hasattr(train_state, "data_state_dict") and not hasattr(train_state, "batch_sampler"):
                # V1
                ckpt_mm.train_dl.dataset.load_state_dict(
                    llm_load(os.path.join(load_ckpt_folder, "sampler_0.pt")), ckpt_path=load_ckpt_folder
                )
                load_content_str += f"{CheckpointLoadContent.SAMPLER}, "
            else:
                if gpc.is_rank_for_log():
                    logger.warning(
                        "CheckpointManager has no 'data_state_dict', skip reload data_state_dict checkpoint!"
                    )

    return load_content_str


class CheckpointLoadMethod:
    """The registration class of the checkpoint loading method,
    users can define their own custom ckpt loading methods."""

    LOAD_FUNC_SIG = None
    LOAD_TYPE_FUNC = {"internevo": try_load_internevo_ckpt}

    @staticmethod
    def register_ckpt_load_type(load_type: str, load_func: Callable):
        """Register"""
        if load_type in CheckpointLoadMethod.LOAD_TYPE_FUNC and gpc.is_rank_for_log():
            logger.warning(f"{load_type} has already been registered!")
            return

        CheckpointLoadMethod.LOAD_TYPE_FUNC.update({load_type: load_func})

    @staticmethod
    def get_ckpt_load_type_func(load_type: str):
        return CheckpointLoadMethod.LOAD_TYPE_FUNC[load_type]


class CheckpointLoadMask:
    """
    According to the content field in the incoming ckpt_info, decide which components to load.
    """

    LOAD_CONTENT_DICT = {
        "model": CheckpointLoadContent.MODEL,
        "sampler": CheckpointLoadContent.SAMPLER,
        "optimizer": CheckpointLoadContent.OPIMIZER,
        "scheduler": CheckpointLoadContent.SCHEDULAER,
    }

    def __init__(self, content: tuple) -> None:
        self.load_set = set(map(lambda x: x.lower(), content))
        if "all" in self.load_set:
            self.load_set = set(CheckpointLoadMask.LOAD_CONTENT_DICT.values())
        else:
            self.load_set = set(map(lambda x: CheckpointLoadMask.LOAD_CONTENT_DICT[x.lower()], content))

    def need_load(self, content: CheckpointLoadContent):
        return content in self.load_set

    def not_only_load(self, content: CheckpointLoadContent):
        return content in self.load_set and len(self.load_set) > 1

    def only_load(self, content: CheckpointLoadContent):
        return set((content,)) == self.load_set

    def __str__(self) -> str:
        return f"{self.load_set}."

    def __repr__(self) -> str:
        return f"{self.load_set}."


def try_load_sensenovalm_ckpt_func(ckpt_mm, load_info, *args, func=None, **kwargs):  # pylint: disable=W0613
    load_content_str = ""
    load_ckpt_folder = load_info["path"]
    if gpc.is_rank_for_log():
        logger.info(f"Try load_ckpt_folder: {load_ckpt_folder}")

    assert func is not None, "Please specify a loading type by setting `ckpt_type` of LOAD_CKPT_FOLDER_INFO in configs"
    func(folder=load_ckpt_folder, model=ckpt_mm.model)

    load_content_str += f"{CheckpointLoadContent.MODEL}, "
    sensenovalm_accelerator.synchronize()

    if isinstance(ckpt_mm.optimizer, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        ckpt_mm.optimizer.reload_zero_fp32_buff()


class CheckpointManager:
    """StorageManagerContext"""

    def __init__(
        self,
        ckpt_config,
        model,
        train_dl=None,
        optimizer=None,
        lr_scheduler=None,
        model_config=None,
        model_config_file=None,
        feishu_address=None,
    ) -> None:
        """
        CheckpointManager is used to decide when to store ckpt. If it is an asynchronous
        upload mode, you must call wait_async_upload_finish at the end of the program to wait
        for the asynchronous ckpt upload to complete.

        Args:
            ckpt_config (dict): model checkpoint config.
            model (nn.module): model obj.
            optimizer (object): optimizer obj.
            lr_scheduler (object): lr_scheduler obj.
            model_config (dict): model config.
        """
        self.enable_save_ckpt = get_config_value(ckpt_config, "enable_save_ckpt", False)
        self.checkpoint_every = get_config_value(ckpt_config, "checkpoint_every", 100)
        self.save_ckpt_folder = get_config_value(ckpt_config, "save_ckpt_folder", None)
        self.oss_snapshot_freq: int = get_config_value(ckpt_config, "oss_snapshot_freq", 50)
        self.stop_file_path = get_config_value(ckpt_config, "stop_file_path", None)
        if self.save_ckpt_folder:
            self.snapshot_ckpt_folder = get_config_value(
                ckpt_config, "snapshot_ckpt_folder", os.path.join(self.save_ckpt_folder, "snapshot")
            )
            self.async_upload_tmp_folder = get_config_value(
                ckpt_config, "async_upload_tmp_folder", "/dev/shm/sensenovalm_tmp_ckpt/"
            )
        else:
            self.snapshot_ckpt_folder = None
            self.async_upload_tmp_folder = None

        self.async_upload = get_config_value(ckpt_config, "async_upload", False)
        self.enable_internevo2hf_ckpt = get_config_value(ckpt_config, "enable_internevo2hf_ckpt", False)

        use_processpool = self.save_ckpt_folder is not None and (
            self.save_ckpt_folder.startswith("volc:") or self.save_ckpt_folder.startswith("oss2:")
        )
        # initialization storage manager
        init_storage_manager(self.enable_save_ckpt, self.async_upload_tmp_folder, self.async_upload, use_processpool)

        self.feishu_address = feishu_address
        self.storage_manager = get_storage_manager()
        self.snapshot_counter = -1

        if hasattr(model, "model"):
            model = model.model

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dl = train_dl
        self.model_config = model_config
        self.model_config_file = model_config_file
        self.averaged_model = None

        # Register pre-defined ckpt load type.
        self.defalut_load_type_func = {
            k: partial(try_load_sensenovalm_ckpt_func, func=v) for k, v in LOAD_FUNC_DICT.items()
        }
        # Register huggingface ckpt load type
        if isinstance(model, BaseModel):
            self.defalut_load_type_func.update(
                {
                    "hf": partial(
                        try_load_sensenovalm_ckpt_func,
                        func=model_initializer.get_module(module_name=gpc.config.model_type).load_hf_weights,
                    )
                }
            )

        for ckpt_load_type, func in self.defalut_load_type_func.items():
            CheckpointLoadMethod.register_ckpt_load_type(ckpt_load_type, func)

        # Init alter file.
        if self.stop_file_path and gpc.get_global_rank() == 0:
            dir_path = os.path.dirname(self.stop_file_path)
            if dir_path != "" and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self.stop_file_path, "w", encoding="utf-8") as f:
                f.write("0")

        self.load_ckpt_info = get_config_value(ckpt_config, "load_ckpt_info", None)
        if self.load_ckpt_info is None:  # (legacy): Try Compatible with old interfaces
            self.load_ckpt_info = ckpt_info_sanity_check(ckpt_config)

        # Auto-reload latest checkpoint, it will overwrite the setting of 'load_ckpt_info'.
        self.auto_resume = get_config_value(ckpt_config, "auto_resume", None)
        if self.auto_resume is None:  # (legacy): Try Compatible with old interfaces
            self.auto_resume = auto_resume_sanity_check(ckpt_config)
        if self.auto_resume:
            self.load_ckpt_info = self.query_lastest_ckpt()

        if self.stop_file_path is None and gpc.is_rank_for_log():
            logger.warning("no set stop_file_path, quit_signal_handler is disable")

        # convert to internal representation
        if self.load_ckpt_info:
            assert (
                "path" in self.load_ckpt_info
                and "content" in self.load_ckpt_info
                and "ckpt_type" in self.load_ckpt_info
            ), "please set content in ckpt setting, eg: ckpt = dict(path='', content=['model'], ckpt_type='internevo')"

            if len(self.load_ckpt_info["content"]) > 1 and "model" in self.load_ckpt_info["content"]:
                assert (
                    self.load_ckpt_info["ckpt_type"] == "internevo"
                ), "Only 'internevo' ckpt supports loading states other than 'model' !"

            # replace load_ckpt
            self.load_ckpt_info["content"] = CheckpointLoadMask(self.load_ckpt_info["content"])

        torch.distributed.barrier()
        # test storage setting is ok.
        if self.enable_save_ckpt:
            self.try_ping_storage()

    def quit_signal_handler(self, train_state) -> bool:
        """
        Exit signal detection function, if we write the exit step in the 'QUIT_FILE_PATH' file,
        all ranks will save ckpt and exit.
        Negative integer step means save ckpt.
        Positive integer step means save ckpt and quit.

        Args:
            train_state (TrainState):
        Returns:
            bool: whether to quit.
        """
        now_break, now_save_ckpt, save_type = False, False, CheckpointSaveType.NORMAL_CHECKPOINT

        if self.stop_file_path is None:
            return now_break, now_save_ckpt, save_type

        with torch.no_grad():
            action_step_t = torch.zeros((1,), dtype=torch.int64).to(get_current_device())
            if gpc.get_global_rank() == 0:
                with open(self.stop_file_path, "r+", encoding="utf-8") as f:
                    f.seek(0)
                    msg = f.read()
                    action_step_t.fill_(int(msg))

            torch.distributed.broadcast(action_step_t, src=0)
            action_step = action_step_t.item()
            del action_step_t

        if action_step < 0 and abs(action_step) == train_state.step_count:
            now_save_ckpt = True

        if action_step > 0 and action_step == train_state.step_count:
            now_break, now_save_ckpt = True, True

        return now_break, now_save_ckpt, save_type

    def is_now_to_save_ckpt(self, train_state, force=False) -> (bool, CheckpointSaveType, bool):
        save_ckpts, save_type, now_break = False, CheckpointSaveType.NORMAL_CHECKPOINT, False
        if force:
            return True, CheckpointSaveType.NORMAL_CHECKPOINT, False

        if (
            self.oss_snapshot_freq > 1
            and train_state.step_count > 0
            and train_state.step_count % self.oss_snapshot_freq == 0
        ):
            save_ckpts, save_type = True, CheckpointSaveType.SNAPSHOT_CHECKPOINT

        if (
            train_state.step_count > 0
            and train_state.step_count % self.checkpoint_every == 0
            or train_state.step_count == train_state.total_steps
        ):
            save_ckpts, save_type = True, CheckpointSaveType.NORMAL_CHECKPOINT

        now_break, singal_save_ckpts, singal_save_type = self.quit_signal_handler(train_state)
        if save_ckpts is False:
            save_ckpts = singal_save_ckpts
            save_type = singal_save_type

        return save_ckpts, save_type, now_break

    def try_save_checkpoint(self, train_state, sensenovavl_dl_custom_infos, force=False):
        if not self.enable_save_ckpt:
            return False

        save_ckpts, save_type, now_break = self.is_now_to_save_ckpt(train_state, force=force)

        if save_ckpts:
            save_hf_ckpt_folder = None

            # Wait for the previous round of asynchronous upload storage to complete.
            self.storage_manager.wait()

            # submit all broadcast when save ckpts.
            self.optimizer.submit_bcast_async(submit_all=True)

            if save_type == CheckpointSaveType.SNAPSHOT_CHECKPOINT:
                # Snapshot number, with only two snapshots written alternately.
                self.snapshot_counter = (self.snapshot_counter + 1) % 2
                save_ckpt_folder = os.path.join(self.snapshot_ckpt_folder, f"{self.snapshot_counter}")
            else:
                save_ckpt_folder = os.path.join(self.save_ckpt_folder, str(train_state.step_count))
                save_hf_ckpt_folder = os.path.join(self.save_ckpt_folder, f"{str(train_state.step_count)}_hf")

            # sensenovavl only support save local currently
            assert save_ckpt_folder.startswith("local:")
            save_ckpt_folder = save_ckpt_folder[len("local:") :]

            # sensenovavl save dataset state
            os.makedirs(os.path.join(save_ckpt_folder, "ds_state"), exist_ok=True)
            save_path = os.path.join(save_ckpt_folder, "ds_state", f"ds_state_{gpc.get_global_rank()}.pt")
            torch.save(train_state.ds_state, save_path)

            info_save_path = os.path.join(save_ckpt_folder, "ds_state", f"ds_info_{gpc.get_global_rank()}.pt")
            torch.save(sensenovavl_dl_custom_infos, info_save_path)

            step_path = os.path.join(save_ckpt_folder, "ds_state", f"{train_state.step_count}.step")
            if gpc.get_global_rank() == 0:
                with open(step_path, "w") as file:
                    pass
            torch.distributed.barrier()

            self.save_checkpoint(
                folder=save_ckpt_folder,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                train_state=train_state,
                model_config=self.model_config,
                model_config_file=self.model_config_file,
            )

            if gpc.get_global_rank() == 0 and self.enable_internevo2hf_ckpt and save_hf_ckpt_folder is not None:
                convert_internevo_ckpt_to_hf(src=save_ckpt_folder, tgt=save_hf_ckpt_folder)

            if gpc.get_global_rank() == 0:
                if hasattr(gpc, "tokenizer") and gpc.tokenizer is not None:
                    logger.info("Start saving tokenizer ...")
                    tokenizer = gpc.tokenizer
                    tokenizer.save_pretrained(save_ckpt_folder)

            if (
                isinstance(self.model, BaseModel)
                and self.enable_internevo2hf_ckpt
                and save_type == CheckpointSaveType.NORMAL_CHECKPOINT
                and gpc.is_rank_for_log()
            ):
                # convert internevo2hf checkpoint
                save_hf_ckpt_folder = os.path.join(self.save_ckpt_folder, f"{str(train_state.step_count)}_hf")
                logger.info(
                    f"Start to convert internevo2hf checkpoint from {save_ckpt_folder} to {save_hf_ckpt_folder}."
                )
                model_initializer.get_module(module_name=gpc.config.model_type).convert_internevo2hf_weights(
                    src=save_ckpt_folder, tgt=save_hf_ckpt_folder
                )
                logger.info(
                    f"Finish to convert internevo2hf checkpoint from {save_ckpt_folder} to {save_hf_ckpt_folder}."
                )
            torch.distributed.barrier()

        return now_break

    def wait_async_upload_finish(self):
        """wait for all checkpoint uploads to be completed"""
        self.storage_manager.wait()
        torch.distributed.barrier()

    def query_latest_snapshot_step_boto3(self):
        """Query the latest snapshot step from the storage backend.
        Currently, we only support the following storage backends: boto3, oss2 and volc.
        Returns:
            Tuple(str, int): path of latest ckpt and ckpt step, if not found, None will return.
        """
        ckpt_list = self.storage_manager.get_fns(self.save_ckpt_folder)
        if ckpt_list is None or len(ckpt_list) == 0:
            return None, None

        max_normal_step = 0
        # Return ckpt_list look like: ['pings', 'snapshot', '4']
        # Here we only try to find the ckpt folder named after step, ignoring snapshot and other folders.
        ckpt_list = [int(fn.strip("/")) for fn in ckpt_list if fn.strip("/").isdigit()]
        if len(ckpt_list) == 0:
            if gpc.is_rank_for_log():
                logger.warning("No available normal checkpoint found. Check your checkpoint path.")
        else:
            if gpc.is_rank_for_log():
                logger.info(f"Found available normal checkpoint: {ckpt_list}")

            ckpt_list.sort(reverse=True)
            for ckpt in ckpt_list:
                fns_list = self.storage_manager.get_fns(os.path.join(self.save_ckpt_folder, str(ckpt)))
                for fn in fns_list:
                    if fn.endswith(".step"):
                        max_normal_step = ckpt
                        break
                if max_normal_step != 0:
                    break

                max_normal_step = ckpt_list[0]
            load_normal_ckpt_path = os.path.join(self.save_ckpt_folder, str(max_normal_step))

        snapshot_path_0 = os.path.join(self.save_ckpt_folder, "snapshot", "0")
        snapshot_path_1 = os.path.join(self.save_ckpt_folder, "snapshot", "1")
        ckpt_list_0 = self.storage_manager.get_fns(snapshot_path_0)
        ckpt_list_1 = self.storage_manager.get_fns(snapshot_path_1)

        def found_latest_snapshot(_ckpt_list):
            _max_step_snapshot = 0
            if _ckpt_list:
                for ckpt in _ckpt_list:
                    ckpt = ckpt.strip("/")
                    if ckpt.endswith(".step"):
                        _max_step_snapshot = max(_max_step_snapshot, int(ckpt.split(".")[0]))
            return _max_step_snapshot

        max_step_0 = found_latest_snapshot(ckpt_list_0)
        max_step_1 = found_latest_snapshot(ckpt_list_1)

        if sum([max_step_0, max_step_1, max_normal_step]) == 0:
            return None, None
        else:
            snap_load_path = snapshot_path_0 if max_step_0 > max_step_1 else snapshot_path_1
            snap_step = max(max_step_0, max_step_1)
            load_path = snap_load_path if snap_step > max_normal_step else load_normal_ckpt_path
            return load_path, max(snap_step, max_normal_step)

    def query_latest_snapshot_step_local(self):
        """Query the latest snapshot step from the local file system."""
        max_step, max_step_path = 0, None
        save_ckpt_folder = self.save_ckpt_folder.split(":")[1]
        for root, _, files in os.walk(save_ckpt_folder, followlinks=True):
            for fn in files:
                fn = fn.strip("/")
                if fn.endswith(".step") and not root.endswith("ds_state"):
                    # We assume that both sensenovalm ckpt and snapshot ckpt will store the '.step' file
                    # as an integrity flag.
                    step = int(fn.rsplit(".", maxsplit=1)[0])
                    if max_step < step:
                        max_step = step
                        max_step_path = root

        return max_step_path, max_step

    def query_lastest_ckpt(self):
        """Query the latest ckpt via the storage backend."""
        latest_ckpt, step = None, -1
        # Training was automatically restarted by the process, forcing the latest snapshot to be read.
        if self.save_ckpt_folder:
            backend, _ = try_get_storage_backend(self.save_ckpt_folder)
            if backend in ["boto3", "oss2", "volc"]:
                latest_ckpt, step = self.query_latest_snapshot_step_boto3()
            elif backend == "local":
                latest_ckpt, step = self.query_latest_snapshot_step_local()
            else:
                raise NotImplementedError(
                    f"Unsupported backend: {backend}, " "Currently only support `boto3`, `oss2`, `volc` and `local`"
                )

            if latest_ckpt and not latest_ckpt.startswith(backend + ":"):
                latest_ckpt = ":".join([backend, latest_ckpt])

        if gpc.is_rank_for_log():
            logger.info(f"Found latest ckpt {latest_ckpt if latest_ckpt else 'None'}, step: {step}...")

        return dict(path=latest_ckpt, content=("all",), ckpt_type="internevo")

    def try_resume_training(self, train_state: TrainState, current_time=""):
        if self.load_ckpt_info is None or self.load_ckpt_info["path"] is None:
            if gpc.is_rank_for_log():
                logger.info(
                    f"===========New Run {current_time} on host:{socket.gethostname()},rank={gpc.get_global_rank()},"
                    f"tp={gpc.get_local_rank(ParallelMode.TENSOR)},pp={gpc.get_local_rank(ParallelMode.PIPELINE)},"
                    f"dp={gpc.get_local_rank(ParallelMode.DATA)}==========="
                )
        else:
            load_path = self.load_ckpt_info["path"]
            load_content = self.load_ckpt_info["content"]
            load_type = self.load_ckpt_info["ckpt_type"]

            load_func = CheckpointLoadMethod.get_ckpt_load_type_func(load_type)
            load_content_str = load_func(self, self.load_ckpt_info, train_state)

            # If we only load model weight, we need rewrite zero optim's fp32 buffer.
            if (
                "optimizer" not in load_content.load_set
                and isinstance(self.optimizer, (HybridZeroOptimizer, HybridZeroOptimizer_v2))
            ) or gpc.config.get("only_load_lr", False):
                self.optimizer.reload_zero_fp32_buff()

            if gpc.is_rank_for_log():
                logger.info(f"load_ckpt_info : {self.load_ckpt_info}")
                logger.info(
                    f"===========Resume training from `{load_path}` {current_time} on host:"
                    f"{socket.gethostname()}==========="
                )
                if load_content_str:
                    logger.info(f"===========Load contents are: {load_content_str}")

    def set_averaged_model(self, averaged_model) -> None:
        """
        Explicitly bind averaged model to checkpoint manager.
        This is intentionally a named method (instead of a bare assignment) to make debugging easier.
        """
        self.averaged_model = averaged_model
        if gpc.is_rank_for_log():
            logger.info(
                f"[AveragedModel] CheckpointManager bind averaged_model which is not None: {averaged_model is not None}"
            )

    def try_load_averaged_model_from_checkpoint(self):
        """
        If averaged model is enabled and `<load_ckpt>/averaged_model/` exists,
        load averaged model weights into `self.averaged_model`.
        Safe: does not overwrite current training weights.
        """
        if self.averaged_model is None:
            if gpc.is_rank_for_log():
                logger.info("[AveragedModel] Skip resume: ckpt_manager.averaged_model is None")
            return
        if self.load_ckpt_info is None or self.load_ckpt_info.get("path") is None:
            if gpc.is_rank_for_log():
                logger.info("[AveragedModel] Skip resume: no load_ckpt_info/path")
            return

        load_path = self.load_ckpt_info["path"]
        # strip backend prefix for local: path
        if load_path.startswith("local:"):
            load_path = load_path[len("local:") :]
        averaged_model_folder = os.path.join(load_path, "averaged_model")
        if not os.path.exists(averaged_model_folder):
            if gpc.is_rank_for_log():
                logger.info(f"[AveragedModel] Skip resume: averaged_model folder not found: {averaged_model_folder}")
            return

        # Load averaged model weights shard(s) for this rank and initialize averaged_model
        self.averaged_model.load_from_checkpoint_folder(
            self.model,
            averaged_model_folder,
            state_dict_loader=load_model_state_dict,
        )
        if gpc.is_rank_for_log():
            logger.info(f"[AveragedModel] Resumed averaged model weights from {averaged_model_folder}")

    @llm_timeout(func_name="save_checkpoint")
    def save_checkpoint(
        self,
        folder,
        model,
        optimizer,
        scheduler,
        train_state: TrainState,
        model_config: Dict = None,
        model_config_file: str = None,
    ):
        """
        Save checkpoint to the given folder path.
        """

        start = time.time()
        self.set_save_folder(folder, train_state.step_count)
        sensenovalm_accelerator.synchronize()
        torch.distributed.barrier()
        if gpc.is_rank_for_log():
            logger.info(f"Saving checkpoint to `{folder}` at batch count:{train_state.step_count}...")

        timer("save-model").start()
        save_model_checkpoint(folder=folder, model=model)
        timer("save-model").stop()

        # Save averaged model weights (optional) into a sub-folder: <folder>/averaged_model/
        # This keeps backward compatibility with existing ckpt layout.
        if self.averaged_model is not None:
            averaged_model_folder = os.path.join(folder, "averaged_model")
            os.makedirs(averaged_model_folder, exist_ok=True)
            averaged_model_states = self.averaged_model.get_averaged_model_state_dict(model)
            timer("save-averaged-model").start()
            save_model_checkpoint(folder=averaged_model_folder, model=model, state_dict_override=averaged_model_states)
            if gpc.get_local_rank(ParallelMode.TENSOR) == 0 and gpc.get_local_rank(ParallelMode.PIPELINE) == 0 and gpc.is_rank_for_log():
                self.averaged_model.save_to_averaged_model_info(averaged_model_folder)
            timer("save-averaged-model").stop()
            if gpc.is_rank_for_log():
                logger.info(f"Saved averaged model weights to `{averaged_model_folder}`")
                
        timer("save-optimizer").start()
        save_optimizer_checkpoint(optim=optimizer, state_path=folder)
        timer("save-optimizer").stop()

        if (
            hasattr(train_state, "data_state_dict")
            and gpc.get_local_rank(ParallelMode.TENSOR) == 0
            and gpc.get_local_rank(ParallelMode.PIPELINE) == 0
        ):
            llm_save(
                os.path.join(folder, f"sampler_{gpc.get_local_rank(ParallelMode.DATA)}.pt"),
                saved_obj=train_state.data_state_dict,
            )

        if gpc.is_rank_for_log():
            if scheduler:
                scheduler_states = scheduler.state_dict()
                llm_save(os.path.join(folder, "schedulder.pt"), saved_obj=scheduler_states)

            if hasattr(train_state, "batch_sampler") and not isinstance(
                train_state.batch_sampler, torch.utils.data.sampler.BatchSampler
            ):
                sampler_state = train_state.batch_sampler.state_dict()
                llm_save(os.path.join(folder, "sampler.pt"), saved_obj=sampler_state)
            llm_save(os.path.join(folder, "context.pt"), saved_obj=train_state.state_dict())

            if model_config is not None:
                # Model configuration dictionary.
                llm_save(os.path.join(folder, "model_config.pt"), saved_obj=model_config)

            if model_config_file is not None:
                # The complete training config file content, stored in binary format.
                llm_save(os.path.join(folder, "config_file.pt"), saved_obj=gpc.config)

        torch.distributed.barrier()

        if gpc.is_rank_for_log():
            timer.log(["save-model", "save-optimizer"], logger=logger)
            logger.info(f"Step: {train_state.step_count}, rank 0 save ckpt use {time.time() - start:.3f} s")
            if self.storage_manager.async_mode is False:
                llm_save(
                    os.path.join(folder, f"{train_state.step_count}.step"),
                    saved_obj=dict({"step": train_state.step_count}),
                )

    def set_save_folder(self, folder, step):
        self.storage_manager.latest_save_folder = folder
        self.storage_manager.latest_save_step = step

    def try_ping_storage(self):
        if gpc.is_rank_for_log():
            buff = torch.ones((1, 64, 64), dtype=torch.bfloat16)
            test_fn = os.path.join(self.save_ckpt_folder, f"pings/{socket.gethostname()}.ping")
            self.storage_manager.save(test_fn, buff)
            self.storage_manager.wait()
            self.storage_manager.load(test_fn)
            del buff
