#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

"""
A .bin file corresponds to a Dataset instance here.
"""

import json
import mmap
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


def gen_shm_meta_name_without_scalar(path: str):
    """gen_shm_meta_name_without_scalar
    Args:
        path (str): dataset path, like:
        /llm_data/tokenized/train/cn/train-00000.bin
    """
    bin_path = Path(path)
    shm_prefix_path = Path(gpc.config.data.shm_path)

    # Use the entire path as the relative path part
    dataset_base_path = bin_path.relative_to(bin_path.anchor)  # Removes the root part (e.g., '/' or 'C:\')

    # /dev/shm/metacache/llm_data/tokenized/train/cn/train-00000.bin
    shm_path_without_num_tokens = Path(shm_prefix_path, dataset_base_path)
    return shm_path_without_num_tokens


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(
        self, path: str, dataset_type_id: int = 0, dataset_subset_id: int = 0, min_length=50, pack_sample_into_one=False
    ):
        self.subset_id = dataset_subset_id
        if not hasattr(gpc.config.data, "use_shm") or not gpc.config.data.use_shm:
            self._process_init(path, dataset_type_id, min_length)
        else:
            devices_per_node = sensenovalm_accelerator.device_count()
            self.local_rank = gpc.get_global_rank() % devices_per_node
            shm_path_without_num_tokens = gen_shm_meta_name_without_scalar(path)

            found_cache, shm_path, num_tokens, seed = False, None, None, None
            while not found_cache:
                if shm_path_without_num_tokens.parent.exists():
                    for file in shm_path_without_num_tokens.parent.iterdir():
                        fp_str = str(file.resolve())
                        if fp_str.startswith(str(shm_path_without_num_tokens.resolve())) and fp_str.endswith(".final"):
                            # Found cache
                            scalers = fp_str.split("%")
                            num_tokens = int(scalers[1])
                            seed = int(scalers[2].split(".")[0])
                            found_cache = True
                            shm_path = fp_str

                # for local_rank 0, no need to wait
                # go forward to do computing and saving
                if self.local_rank == 0:
                    break

                if not found_cache:
                    logger.warning(f"GPU {self.local_rank} loading meta: cache not found, waiting...")
                    time.sleep(1)

            if found_cache:
                assert shm_path and num_tokens is not None and seed is not None
                self.shm_handler = np.load(shm_path, mmap_mode="r+")
                self.offsets = self.shm_handler[0]
                self.lengths = self.shm_handler[1]
                if pack_sample_into_one:
                    self.indices = self.shm_handler[2]
                    self.cum_lens = self.shm_handler[3]
                else:
                    self.sample_indices = self.shm_handler[2]
                    self.len_samples_shuffled = self.shm_handler[3]
                    self.acm_len_samples = self.shm_handler[4]
                self.num_tokens = num_tokens
                self.seed = seed
                self.threadlocal = threading.local()
                self.path = path
                self.resolved_path = Path(path).resolve()
                self.type_id = dataset_type_id
                self.old_length = len(self.offsets)
            elif self.local_rank == 0:
                self._process_init(path, dataset_type_id, min_length)
            else:
                assert False, "should not arrive here"

            self.found_cache = found_cache

    def _process_init(self, path: str, dataset_type_id: int = 0, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f"{resolved_path}.meta")
        self.type_id = dataset_type_id

        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.meta), f"The cache file:{self.meta} is not found for file:{self.path}"
        try:
            with open(self.meta, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {self.meta}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]

        if min_length > 0:
            mask = self.lengths >= min_length
            self.old_lengths = self.lengths.copy()
            self.old_length = len(self.offsets)
            self.offsets = self.offsets[mask]
            self.lengths = self.lengths[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
            item["type_id"] = self.type_id
            item["subset_id"] = self.subset_id
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)
