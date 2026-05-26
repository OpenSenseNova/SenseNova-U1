# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import os
import re
from pathlib import Path

import torch

from sensenovalm.core.context import global_context as gpc


def have_data_files(path):
    for _, _, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".bin"):
                return True
    return False


def get_dataset_type_ids_map(path):
    dirlist = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path) and have_data_files(folder_path):
            dirlist.append(folder_name)
    dirlist.sort()
    return {key: idx for idx, key in enumerate(dirlist)}


def get_dataset_subset_ids_map(path):
    lang_dirlist = list(os.listdir(path))
    lang_dirlist.sort()
    dirlist = []
    for lang in lang_dirlist:
        if not Path(os.path.join(path, lang)).is_dir():
            continue
        _subset = os.listdir(os.path.join(path, lang))
        if len(_subset) == 0:
            continue
        first_file_in_lang_dir = _subset[0]
        _path = Path(os.path.join(path, lang, first_file_in_lang_dir))
        if not _path.is_dir() and (
            first_file_in_lang_dir.endswith(".jsonl") or first_file_in_lang_dir.endswith(".bin")
        ):  # train_folder/lang/xxx.jsonl
            dirlist.append(os.path.join(lang, lang))
        else:  # train_folder/lang/subset/xxx.jsonl
            subset_dirlist = [os.path.join(lang, subset) for subset in os.listdir(os.path.join(path, lang))]
            dirlist.extend(subset_dirlist)
    dirlist.sort()
    return {key: idx for idx, key in enumerate(dirlist)}


def get_dataset_type_id(dataset_type_ids_map, path):
    match_idxes = []

    for key, idx in dataset_type_ids_map.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {dataset_type_ids_map}"
    return match_idxes[0]


def get_dataset_subset_id(dataset_subset_ids_map, path):
    match_idxes = []

    for key, idx in dataset_subset_ids_map.items():
        lang, subset = key.split(os.path.sep)
        if lang == subset:
            if re.search(rf"/[z_]*{lang}/", path):  # folder is lang/xxx.json
                match_idxes.append(idx)
        else:
            if re.search(rf"/[z_]*{key}/", path):  # folder is lang/subset/xxx.json and subset != lang
                match_idxes.append(idx)
    assert (
        len(match_idxes) == 1
    ), f"{path}, match_idxes should be 1, but got {match_idxes} from {dataset_subset_ids_map}"
    return match_idxes[0]


def get_lang_subset_types(train_folder: str = None):
    """Get language and subset types from train_folder.

    If train_folder is None, return default language and subset types.
    """
    if train_folder is not None:
        language_types = list(get_dataset_type_ids_map(train_folder).keys())
        subset_types = list(get_dataset_subset_ids_map(train_folder).keys())
    else:
        language_types = ["en", "cn", "code"]
        subset_types = ["en/subset1", "cn/subset1", "code/subset1"]

    return language_types, subset_types


def _unpack_data(data, cu_seqlens, padding_v: int = 0):
    bsz = data.shape[0]

    num_seq = gpc.config.data["micro_bsz"]
    seq_len_ = gpc.config.data.seq_len
    dtype_ = data.dtype

    outputs = torch.empty(bsz, num_seq, seq_len_, device=data.device, dtype=dtype_).fill_(padding_v)

    for i in range(bsz):
        output = torch.empty(num_seq, seq_len_, device=data.device, dtype=dtype_).fill_(padding_v)
        cu_seqlens_slice = cu_seqlens[i]
        for j in range(num_seq):
            length = cu_seqlens_slice[j + 1] - cu_seqlens_slice[j]
            output[j, 0:length] = data[i, cu_seqlens_slice[j] : cu_seqlens_slice[j + 1]]
        outputs[i] = output

    return outputs


def unpack_type_ids(type_ids, cu_seqlens):
    return _unpack_data(type_ids, cu_seqlens)


def unpack_data(data, label):
    data["input_ids"] = _unpack_data(data["input_ids"], data["cu_seqlens"], padding_v=0).squeeze(0)
    data["indexes"] = _unpack_data(data["indexes"], data["cu_seqlens"], padding_v=0).squeeze(0)
    label = _unpack_data(label, data["cu_seqlens"], padding_v=-100).squeeze(0)

    data["max_seqlen"] = gpc.config.data.seq_len

    data.pop("cu_seqlens")
    # indexes will be used in rotary emb when using isp and sp_size > 1
    # data.pop("indexes")
    # per batch's index should be equal, so we select first batch
    data["indexes"] = data["indexes"][0]

    # If model has inject_info and data_helper is enabled, we provide position_ids
    if "inject_info" in gpc.config.model and gpc.config.model.inject_info.get("data_helper", False):
        data.pop("max_seqlen")
        data["position_ids"] = data.pop("indexes").unsqueeze(0)  # [batch, seqlen]

    return data, label


def packed_data_normalizer(data, label):
    # Should we normalize packed data in this form of this data processor
    # or let the dataset handle it? Currently inclined towards the latter.
    assert data["input_ids"].shape[0] == 1, "data should be packed with batch size 1"

    data["indexes"] = data["indexes"][0]
    data["cu_seqlens"] = data["cu_seqlens"][0].squeeze(0)
    data["max_seqlen"] = (data["cu_seqlens"][1:] - data["cu_seqlens"][:-1]).max().item()

    # If model has inject_info and data_helper is enabled, we provide position_ids, cu_seqlens, max_seqlen
    if "inject_info" in gpc.config.model and gpc.config.model.inject_info.get("data_helper", False):
        gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"] = data.pop("cu_seqlens")
        gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"] = data.pop("max_seqlen")
        data["position_ids"] = data.pop("indexes").unsqueeze(0)  # [batch, seqlen]

    return data, label
