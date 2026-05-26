# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import gc
import json
import os
import copy

from functools import partial

import torch
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader
from transformers import AutoTokenizer

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.data.build_dataloader import get_tokenized_train_loader_items
from sensenovalm.data.build_dataloader import build_train_loader_with_data_type as pretrain_build_train_loader_with_data_type
from sensenovalm.utils.logger import get_logger
from sensenovavl.data.batch_sampler import LengthGroupedSampler
from sensenovavl.data.constants import (
    ALL_SPECIAL_TOKEN_LIST,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    THINK_START_TOKEN,
    THINK_END_TOKEN
)
from sensenovavl.data.dataset import TCSLoader
from sensenovavl.data.dataset_interleaved_iterable import (
    ImageTextPairDataset,
    InterleavedDataset,
    PackedDataset,
    internevo_collate_fn,
)
from sensenovavl.data.distributed_sampler import DistributedSampler

from sensenovavl.data.multimodal_dataset import (
    build_datasets,
    image_pair_collator,
)
from sensenovalm.data.utils import get_dataset_type_ids_map
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)

# NOTE: all data is assumed to live on the local filesystem; ``TCSLoader`` is a
# thin local image/video loader (object-storage code paths have been removed).


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def maybe_set_torch_sharing_strategy(sharing_strategy):
    if not sharing_strategy:
        return

    try:
        current_strategy = torch.multiprocessing.get_sharing_strategy()
        if current_strategy != sharing_strategy:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
            logger.info(f"Set torch multiprocessing sharing strategy to `{sharing_strategy}`")
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.warning(f"Failed to set torch multiprocessing sharing strategy to `{sharing_strategy}`: {exc}")


def get_multimodal_streaming_train_loader_items(data_cfg):   # NOTE:
    model_cfg = gpc.config.model
    tokenizer_path = data_cfg.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    gpc.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.tokenizer_path = tokenizer_path
    if hasattr(data_cfg, 'max_sample_tokens'):
        tokenizer.model_max_length = data_cfg.max_sample_tokens
    else:
        tokenizer.model_max_length = data_cfg.seq_len
    token_list = ALL_SPECIAL_TOKEN_LIST
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    if gpc.config.get('add_think_tag', False):
        num_new_tokens = num_new_tokens + tokenizer.add_tokens([THINK_START_TOKEN, THINK_END_TOKEN], special_tokens=False)
    num_pad_token = model_cfg.vocab_size - len(tokenizer)
    pad_token_list = [f"<FAKE_PAD_{i}>" for i in range(num_pad_token)]  # padding vocab_size]
    num_new_tokens = num_new_tokens + tokenizer.add_tokens(pad_token_list, special_tokens=True)

    assert num_pad_token >= 0
    assert num_pad_token >= 0
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    # update multimodal model cfg
    # model_cfg._add_item("num_new_tokens", num_new_tokens)
    model_cfg._add_item("img_context_token_id", img_context_token_id)
    model_cfg._add_item("img_start_token_id", img_start_token_id)
    model_cfg._add_item("img_end_token_id", img_end_token_id)
    model_cfg.vit_cfg._add_item("image_size", data_cfg.force_image_size)
    model_cfg.vit_cfg._add_item("patch_size", data_cfg.patch_size)

    assert model_cfg.vocab_size == len(tokenizer)

    data_cfg._add_item("img_context_token_id", img_context_token_id)
    data_cfg._add_item("img_start_token_id", img_start_token_id)
    data_cfg._add_item("img_end_token_id", img_end_token_id)

    logger.info(f"{num_new_tokens=}, {len(tokenizer)=}")

    gpc.tokenizer = tokenizer

    if gpc.get_local_rank(ParallelMode.COMMON_DATA) > 0:
        return None, None, None, None
    
    
    # NLP/streaming data mix removed for OSS; multimodal-only path.
    nlp_mm_sampling_fixed_token = False

    datasets, dataset_weights = [], []
    dataset_types = []

    # multimodal dataset
    num_image_token = int((data_cfg.force_image_size // data_cfg.patch_size) ** 2 * (data_cfg.down_sample_ratio**2))
    tcs_loader = TCSLoader()

    mm_paired_datasets, mm_cc_datasets, mm_plain_pair_datasets, mm_paired_lengths, mm_cc_lengths, mm_plain_pair_lengths, mm_dataset_types = build_datasets(
        data_args=data_cfg,
        tokenizer=tokenizer,
        tcs_loader=tcs_loader,
        num_image_token=num_image_token,
        data_rank=gpc.get_local_rank(ParallelMode.DATA),
        data_world_size=gpc.get_world_size(ParallelMode.DATA),
        distributed_mode=data_cfg.use_packed_ds,
        group_by_length=data_cfg.group_by_length,
        force_shuffle=data_cfg.use_packed_ds,
        dynamic_image_size=data_cfg.dynamic_image_size,
        dynamic_image_version=getattr(data_cfg, "dynamic_image_version", 'v0'),
        use_thumbnail=data_cfg.use_thumbnail,
        min_num_frame=getattr(data_cfg, 'min_num_frame', 4),
        max_num_frame=getattr(data_cfg, 'max_num_frame', 24),
        min_dynamic_patch=data_cfg.min_dynamic_patch,
        max_dynamic_patch=data_cfg.max_dynamic_patch,
        data_augment=getattr(data_cfg, "data_augment", True),
        type_id_offset=len(dataset_types),
    )
    dataset_types.extend(mm_dataset_types)
    datasets.extend(mm_paired_datasets + mm_cc_datasets + mm_plain_pair_datasets)
    paired_total_length = sum(mm_paired_lengths)
    cc_total_length = sum(mm_cc_lengths)
    plain_pair_total_length = sum(mm_plain_pair_lengths)

    mm_total_weight = 1.0 - getattr(data_cfg, 'llm_data_weights', 0.0)
    mm_cc_data_weights = getattr(data_cfg, 'mm_cc_data_weights', 0.0) * mm_total_weight
    mm_plain_pair_data_weights = getattr(data_cfg, 'mm_plain_pair_data_weights', 0.0) * mm_total_weight
    mm_paired_total_weight = mm_total_weight - mm_cc_data_weights - mm_plain_pair_data_weights

    mm_paired_weights = [l / paired_total_length * mm_paired_total_weight for l in mm_paired_lengths]
    mm_cc_weights = [l / cc_total_length * (mm_total_weight- mm_paired_total_weight) for l in mm_cc_lengths]
    mm_plain_pair_weights = [l / plain_pair_total_length * mm_plain_pair_data_weights for l in mm_plain_pair_lengths]

    if gpc.is_rank_for_log():
        logger.info(f'the sampling weight:\n paired: {sum(mm_paired_weights)}\n cc: {sum(mm_cc_weights)}\n plain_pair: {sum(mm_plain_pair_weights)}')
    dataset_weights.extend(mm_paired_weights + mm_cc_weights + mm_plain_pair_weights)

    if data_cfg.use_packed_ds:
        train_ds = PackedDataset(       # NOTE:
            tokenizer=tokenizer,
            data_rank=gpc.get_local_rank(ParallelMode.DATA),
            data_world_size=gpc.get_world_size(ParallelMode.DATA),
            datasets=datasets,
            dataset_weight=dataset_weights,
            num_images_expected=data_cfg.num_images_expected,
            max_packed_tokens=data_cfg.max_packed_tokens,
            max_buffer_size=data_cfg.max_buffer_size,
            log_freq=data_cfg.log_freq,
            strict_mode=data_cfg.strict_mode,
            debug_mode=getattr(data_cfg, "debug_mode", False),
            replacement=getattr(data_cfg, "replacement", False),
            allow_overflow=getattr(data_cfg, "allow_overflow", False),
            allow_empty_data=getattr(data_cfg, "allow_empty_data", False),
            allow_deduplicated_ds_name=False,
            nlp_mm_sampling_fixed_token=nlp_mm_sampling_fixed_token,
            packed_buffer_stale_threshold=getattr(data_cfg, "packed_buffer_stale_threshold", 200),
        )
        train_sampler = None
        train_collate_fn = partial(
            internevo_collate_fn,
            max_item_length=data_cfg.max_packed_tokens,
            img_start_token_id=img_start_token_id,
            img_token_id=img_context_token_id,
            img_end_token_id=img_end_token_id,
            ignored_token_ids=[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("\n")],
            micro_num=data_cfg.micro_num,
            len2weight=partial(len2weight, loss_reduction=getattr(data_cfg, "loss_reduction", "token")),
            loss_reduction_all_gather=getattr(data_cfg, "loss_reduction_all_gather", False),
            patch_size=data_cfg.patch_size,
        )
    else:
        raise NotImplementedError('data_cfg.use_packed_ds is False')   # NOTE:
        train_sampler = LengthGroupedSampler(
            train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
            batch_size=data_cfg.micro_num
            * data_cfg.micro_bsz,  # because multimodal SenseNovaVL-MoE-Chat is non-pack data, we use the same data format
            rampup_batch_size=data_cfg.rampup_batch_size,
            micro_bsz=data_cfg.micro_bsz,
            seed=data_cfg.get("seed", 42),
            drop_last=True,
            data_rank=gpc.get_local_rank(ParallelMode.DATA),
            data_world_size=gpc.get_world_size(ParallelMode.DATA),
        )
        train_collate_fn = partial(
            image_pair_collator,
            max_item_length=data_cfg.max_packed_tokens,
            img_start_token_id=img_start_token_id,
            img_token_id=img_context_token_id,
            img_end_token_id=img_end_token_id,
            ignored_token_ids=[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("\n")],
        )

    return train_ds, train_sampler, train_collate_fn, dataset_types


def get_multimodal_packed_streaming_train_loader_items(data_cfg):
    model_cfg = gpc.config.model
    tokenizer_path = data_cfg.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    gpc.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_cfg.seq_len
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        *[f"<PAD_{i}>" for i in range(7)],  # padding vocab_size
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    # update multimodal model cfg
    # model_cfg._add_item("num_new_tokens", num_new_tokens)
    model_cfg._add_item("img_context_token_id", img_context_token_id)
    model_cfg._add_item("img_start_token_id", img_start_token_id)
    model_cfg._add_item("img_end_token_id", img_end_token_id)
    model_cfg.vit_cfg._add_item("image_size", data_cfg.force_image_size)
    model_cfg.vit_cfg._add_item("patch_size", data_cfg.patch_size)

    assert model_cfg.vocab_size == len(tokenizer)

    data_cfg._add_item("img_context_token_id", img_context_token_id)
    data_cfg._add_item("img_start_token_id", img_start_token_id)
    data_cfg._add_item("img_end_token_id", img_end_token_id)

    logger.info(f"{num_new_tokens=}, {len(tokenizer)=}")

    gpc.tokenizer = tokenizer

    num_image_token = int((data_cfg.force_image_size // data_cfg.patch_size) ** 2 * (data_cfg.down_sample_ratio**2))
    tcs_loader = TCSLoader()

    with open(data_cfg.meta_path) as file:
        ds_collections = json.load(file)

    datasets = []
    dataset_weight = []
    for ds_name in ds_collections.keys():
        if ds_collections[ds_name]["dataset_type"] == "interleaved":
            dataset_type = InterleavedDataset
        elif ds_collections[ds_name]["dataset_type"] == "pair":
            dataset_type = ImageTextPairDataset
        else:
            raise NotImplementedError(ds_collections[ds_name]["dataset_type"])

        datasets.append(
            dataset_type(
                template_name=data_cfg.conv_style,
                meta=ds_collections[ds_name],
                tokenizer=tokenizer,
                tcs_loader=tcs_loader,
                ds_name=ds_name,
                data_rank=gpc.get_local_rank(ParallelMode.DATA),
                data_world_size=gpc.get_world_size(ParallelMode.DATA),
                num_image_token=num_image_token,
                image_size=data_cfg.force_image_size,
                is_train=ds_collections[ds_name]["data_augment"],
                pad2square=data_cfg.pad2square,
                group_by_length=data_cfg.group_by_length,
                dynamic_image_size=data_cfg.dynamic_image_size,
                use_thumbnail=data_cfg.use_thumbnail,
                min_dynamic_patch=data_cfg.min_dynamic_patch,
                max_dynamic_patch=data_cfg.max_dynamic_patch,
                max_num_images=data_cfg.num_images_expected,
                image_switch_prob=ds_collections[ds_name].get("image_switch_prob", 0),
            )
        )
        dataset_weight.append(ds_collections[ds_name]["weight"])
    train_ds = PackedDataset(
        tokenizer=tokenizer,
        data_rank=gpc.get_local_rank(ParallelMode.DATA),
        data_world_size=gpc.get_world_size(ParallelMode.DATA),
        datasets=datasets,
        dataset_weight=dataset_weight,
        num_images_expected=data_cfg.num_images_expected,
        max_packed_tokens=data_cfg.max_packed_tokens,
        max_buffer_size=data_cfg.max_buffer_size,
        log_freq=data_cfg.log_freq,
        strict_mode=data_cfg.strict_mode,
        debug_mode=getattr(data_cfg, "debug_mode", False),
        packed_buffer_stale_threshold=getattr(data_cfg, "packed_buffer_stale_threshold", 200),
    )

    train_sampler = None
    train_collate_fn = partial(
        internevo_collate_fn,
        max_item_length=data_cfg.max_packed_tokens,
        img_start_token_id=train_ds.datasets[0].img_start_token_id,
        img_token_id=train_ds.datasets[0].img_token_id,
        img_end_token_id=train_ds.datasets[0].img_end_token_id,
        ignored_token_ids=[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("\n")],
        micro_num=data_cfg.micro_num,
        patch_size=data_cfg.patch_size,
    )

    return train_ds, train_sampler, train_collate_fn


class MyCustomDataLoader(DataLoader):
    """
    MyCustomDataLoader
    """

    def __init__(self, dataset, *args, batch_size=1, micro_num=1, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.micro_num = micro_num

    def __iter__(self):
        for batch in super().__iter__():
            data, label = batch
            assert isinstance(data, dict) and isinstance(label, torch.Tensor)

            stacked_data = {}
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    stacked_data[key] = torch.cat([data[key]] * self.micro_num, dim=0)
                elif isinstance(data[key], list):
                    assert len(data[key]) == 1
                    stacked_data[key] = [data[key][0] for _ in range(self.micro_num)]
                else:
                    stacked_data[key] = data[key]
            stacked_label = torch.cat([label] * self.micro_num, dim=0)

            yield (stacked_data, stacked_label)


class RestartableDataLoader(DataLoader):
    """DataLoader with explicit worker-pool restart support."""

    def __init__(self, *args, worker_fail_retry=2, worker_fail_retry_backoff_s=3.0, **kwargs):
        self.worker_fail_retry = max(int(worker_fail_retry), 0)
        self.worker_fail_retry_backoff_s = max(float(worker_fail_retry_backoff_s), 0.0)
        super().__init__(*args, **kwargs)

    def restart_workers(self):
        iterator = getattr(self, "_iterator", None)
        if iterator is not None:
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                try:
                    shutdown_workers()
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning(f"Failed to shutdown dataloader workers cleanly: {exc}")
            self._iterator = None

        gc.collect()
        return iter(self)


class FakeDataset:
    def __init__(self) -> None:
        self.name = "fake"

    def load_state_dict(self, *args, **kwargs):
        pass


class FakeDataLoader:
    """
    FakeDataLoader
    """

    def __init__(self):
        self.dataset = FakeDataset()
        self.batch_sampler = None
        self.worker_fail_retry = 0
        self.worker_fail_retry_backoff_s = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        # 注意要和真实数据返回的那个list的size保持一致，比如2个元素 分别是input和label两个dict
        return [{}, {}]  # input and label

    def restart_workers(self):
        return iter(self)


def _build_multimodal_dataloader_kwargs(data_cfg, train_collate_fn):
    num_workers = data_cfg.get("num_workers", 4)
    persistent_workers = data_cfg.get("persistent_workers", False) and num_workers > 0
    sharing_strategy = data_cfg.get("sharing_strategy", os.environ.get("TORCH_SHARING_STRATEGY", "file_system"))

    maybe_set_torch_sharing_strategy(sharing_strategy)

    dataloader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=train_collate_fn,
        persistent_workers=persistent_workers,
        worker_init_fn=partial(worker_init_fn, sharing_strategy=sharing_strategy),
        worker_fail_retry=data_cfg.get("worker_fail_retry", 2),
        worker_fail_retry_backoff_s=data_cfg.get("worker_fail_retry_backoff_s", 3.0),
    )

    if num_workers > 0:
        prefetch_factor = data_cfg.get("prefetch_factor", 1)
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)

        dataloader_timeout = data_cfg.get("dataloader_timeout", 0)
        if dataloader_timeout:
            dataloader_kwargs["timeout"] = float(dataloader_timeout)

        multiprocessing_context = data_cfg.get("multiprocessing_context", None)
        if multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context

    return dataloader_kwargs


def build_train_loader_with_data_type():
    """
    Build and return the training data loader based on data type.

    Returns: A tuple of (train_dl, dataset_types).
    """
    data_cfg = gpc.config.data
    dataset_types = None

    # for pretrain
    if data_cfg.type in ["tokenized", "streaming"]:
        train_dl = pretrain_build_train_loader_with_data_type(
            data_cfg, 
            data_rank=gpc.get_local_rank(ParallelMode.DATA),
            data_world_size=gpc.get_world_size(ParallelMode.DATA)
        )
        return train_dl, dataset_types

    # for sft
    if data_cfg.type in [
        "multimodal_streaming",
        "multimodal_packed_streaming",
        "random_multimodal_packed_streaming",
    ]:
        # dataset_types not used in cuurent multimodal SenseNovaVL-MoE-Chat
        dataset_types = None
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if data_cfg.type == "multimodal_streaming":  # NOTE:
        train_ds, train_sampler, train_collate_fn, dataset_types = get_multimodal_streaming_train_loader_items(data_cfg)
    elif data_cfg.type == "multimodal_packed_streaming":
        train_ds, train_sampler, train_collate_fn = get_multimodal_packed_streaming_train_loader_items(data_cfg)
    elif data_cfg.type == "random_multimodal_packed_streaming":
        train_ds, train_sampler, train_collate_fn = get_tokenized_train_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if gpc.get_local_rank(ParallelMode.COMMON_DATA) > 0:
        train_dl = FakeDataLoader()
        return train_dl, None

    dataloader_kwargs = _build_multimodal_dataloader_kwargs(data_cfg, train_collate_fn)

    train_dl = RestartableDataLoader(      # NOTE:
        dataset=train_ds,
        batch_size=data_cfg.micro_num,
        batch_sampler=train_sampler,
        **dataloader_kwargs,
    )

    return train_dl, dataset_types


def worker_init_fn(worker_id, sharing_strategy=None):  # pylint: disable=W0613
    maybe_set_torch_sharing_strategy(sharing_strategy)
    gc.enable()
