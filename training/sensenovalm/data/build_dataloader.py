# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from functools import partial
from typing import Dict

import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from sensenovalm.data.tokenized.batch_sampler import (
    StaticBatchSampler,
    StaticBatchSamplerWithWeights,
    get_dataset_weights,
    get_dpsampler_dataloader,
)
from sensenovalm.data.tokenized.collaters import jsonl_ds_collate_fn, packed_collate_fn
from sensenovalm.data.tokenized.dataset import get_dataset_dict
from sensenovalm.data.tokenized.dummy_dataset import RandomDataset
from sensenovalm.data.tokenized.packed_dataset import (
    PackedDatasetWithCut,
    PackedDatasetWithoutCuSeqlen,
    get_packed_dataset_without_short_length,
)
from sensenovalm.utils.common import is_rank_for_log
from sensenovalm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


def get_tokenized_train_loader_items(
    data_cfg: Dict,
    data_rank: int,
    data_world_size: int,
):
    """Get the training data loader for tokenized dataset."""
    if data_cfg.get("train_folder", None) is None:
        if is_rank_for_log():
            logger.info("Detect `train_folder` is None, generating random dataset..")

        train_ds = RandomDataset(
            num_samples=1024 * max(data_world_size, 1),
            max_len=data_cfg.seq_len,
            fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
        )

        if data_cfg.pack_sample_into_one:
            train_ds = PackedDatasetWithoutCuSeqlen(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
        else:
            train_ds = PackedDatasetWithCut(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )

        train_sampler = StaticBatchSampler(
            train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
            batch_size=data_cfg.micro_num,
            rampup_batch_size=data_cfg.get("rampup_batch_size", None),
            micro_bsz=data_cfg.micro_bsz,
            seed=data_cfg.get("seed", 1024),
            drop_last=True,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )
        train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    else:
        train_ds = get_packed_dataset_without_short_length(
            folder=data_cfg.train_folder,
            packed_length=data_cfg.packed_length,
            max_length_per_sample=data_cfg.seq_len,
            show_progress=dist.get_rank() == 0,
            min_length=data_cfg.get("min_length", 0),
            min_length_dict=data_cfg.get("min_length_dict", None),
            break_mode=data_cfg.get("break_mode", "cut"),
            pack_sample_into_one=data_cfg.get("pack_sample_into_one", False),
        )

        if hasattr(data_cfg, "dataset_weights") and data_cfg.dataset_weights is not None:
            if isinstance(train_ds, (PackedDatasetWithCut, PackedDatasetWithoutCuSeqlen)):
                datasets = [train_ds]
            else:
                datasets = train_ds.datasets

            if is_rank_for_log():
                logger.info(data_cfg.dataset_weights)

            dataset_weights = get_dataset_weights(data_cfg.dataset_weights, datasets)

            train_sampler = StaticBatchSamplerWithWeights(
                datasets,
                dataset_weights,
                batch_size=data_cfg.micro_num * data_cfg.micro_bsz,  # Compatible with train_llm
                micro_bsz=data_cfg.micro_bsz,
                drop_last=data_cfg.get("drop_last", True),
                rampup_batch_size=data_cfg.get("rampup_batch_size", None),
                seed=data_cfg.get("seed", 1024),
                data_rank=data_rank,
                data_world_size=data_world_size,
            )
            train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

        else:
            train_sampler = StaticBatchSampler(
                train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
                batch_size=data_cfg.micro_num,
                rampup_batch_size=data_cfg.get("rampup_batch_size", None),
                micro_bsz=data_cfg.micro_bsz,
                seed=data_cfg.get("seed", 1024),
                drop_last=True,
                data_rank=data_rank,
                data_world_size=data_world_size,
            )
            train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    return train_ds, train_sampler, train_collate_fn


def get_tokenized_valid_loader_items(data_cfg):
    """Get the validation data loader for tokenized dataset."""
    if not data_cfg.valid_folder:
        if is_rank_for_log():
            logger.info("Detect `valid_folder` is None, generating random dataset..")
        valid_ds = RandomDataset(
            num_samples=4096,
            max_len=data_cfg.seq_len,
            fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
        )
    else:
        valid_ds = get_dataset_dict(folder=data_cfg.valid_folder, split="")

    if not isinstance(valid_ds, dict):
        valid_ds = {"val": valid_ds}

    valid_collate_fn = partial(jsonl_ds_collate_fn, max_length_per_sample=data_cfg.seq_len)

    return valid_ds, valid_collate_fn


def build_train_loader_with_data_type(
    data_cfg: Dict,
    data_rank: int = 0,
    data_world_size: int = 1,
):
    """
    Build and return the training data loader based on data type.

    Returns: A tuple of (train_dl, language_types).
    """
    num_workers = data_cfg.get("num_worker", 4)

    if data_cfg.type == "tokenized":
        train_ds, train_sampler, train_collate_fn = get_tokenized_train_loader_items(
            data_cfg, data_rank=data_rank, data_world_size=data_world_size
        )
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=num_workers > 0,
    )

    return train_dl


def build_valid_loader_with_data_type(
    data_cfg: Dict,
    data_world_size: int = 1,
):
    """Generate and return the validation data loader based on data type."""
    num_workers = data_cfg.get("num_worker", 0)

    if data_cfg.type == "tokenized":
        valid_ds, valid_collate_fn = get_tokenized_valid_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if valid_ds is None:
        return None

    val_dls = {}
    for val_name, ds in valid_ds.items():

                    # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
            # otherwise too much data may be dropped
            batch_size = min(data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // data_world_size)
            batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz

            if batch_size == 0 and is_rank_for_log():
                logger.info(f"skip validate {val_name}.")
                continue

            val_dls[val_name] = get_dpsampler_dataloader(
                ds,
                shuffle=False,
                num_workers=num_workers,
                batch_size=batch_size,
                collate_fn=valid_collate_fn,
                drop_last=True,
            )  # drop_last=True, otherwise it may cause problems in the last batch

            if is_rank_for_log():
                logger.info(
                    f"load validation dataset {val_name} with valid batch size {str(batch_size)} and "
                    f"samples {str(len(val_dls[val_name]))}."
                )

    return val_dls
