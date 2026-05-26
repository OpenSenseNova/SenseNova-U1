# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Uses `LabelSmoother` ignore-index from HuggingFace transformers
# (HuggingFace Inc., Apache-2.0).
import bisect
import copy
import math
import hashlib
import io
import pickle
import json
import os
from collections import defaultdict
from typing import List, Union, Optional

import numpy as np
import torch
import torch.distributed as dist


from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers.trainer_pt_utils import LabelSmoother

import sensenovavl

from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

from .dataset import build_transform, preprocess_pixel_values
from .constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
from .collators import concat_pad_data_collator
import mmap
from io import TextIOWrapper

from sensenovavl.data.dataset import build_transform, dynamic_preprocess_native_resolution

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = get_logger(__name__, logging_level="info")

# 目前的实现逻辑为两层结构，内层读取数据，外层拼接数据
# 其中，内层不设置token数量限制，但是会根据图像数量限制对文档进行切割
# 外层会在此基础上，进一步对token数量进行切割

# 外层的拼接逻辑是维护一个buffer list，每次拿到数据就和list中某个buffer合并
# 如果没有满足条件的buffer，就作为新的buffer直接加入
# 由于引入了chunk attention，内层被切割的文档的不同块都可以等价处理
# 也因为chunk attention的存在，我们可以放心地拼接图文穿插和图文对数据，以保证拼接后数据的图像数量固定

# 目前存在两个缺陷：
# 1. <t1><i1><t2><i2><i3><i4><t3> 可能被拆成 <t1><i1><t2><i2><i3>，后两个<i2><i3>其实没有意义
# 2. 外层虽然保证图像数量，但是不能保证token数量被填满

# TODO: 检查内存泄漏

# TODO: 支持纯文本数据读取


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class BaseDataset(IterableDataset):
    """
    BaseDataset
    """

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        num_image_token,
        ds_name,
        data_rank,
        data_world_size,
        image_size=224,
        is_train=False,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        dynamic_image_version=None,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        max_dynamic_images=4,
        max_multi_image_dynamic_patch=6,
        use_img_start_end_token=True,
        max_num_images=999,
        image_switch_prob=0,
        normalize_type='imagenet',
        type_id=None,
        repeat_time=1,
    ):
        super().__init__()
        data_path = meta['annotation']
        image_path = meta['root']
        dataset_type = meta['dataset_type']

        self.img_path_in_meta = meta.get('img_path_in_meta', True)

        
        self.length = math.ceil(meta['length'] * meta['repeat_time'])
        self.repeat_time = repeat_time
        self.dataset_weight = self.length  if 'length' in meta else meta['weight']

        # for img-text pair data; we usually use the laion original caption
        self.caption_key = meta['caption_key'] if 'caption_key' in meta else 'caption'
        
        min_active_tokens_ratio = meta.get('min_active_tokens_ratio', 1/256)

        self.split_version = meta.get('split_version', 'v1')
        self.permit_pure_text_chunk = meta.get('permit_pure_text_chunk', False)

        self.template_name = template_name
        self.data_path = data_path

        if os.path.isfile(self.data_path):
            self.data_path_is_dir = True
            self.anno_files = [self.data_path]
        elif os.path.isdir(self.data_path):
            self.data_path_is_dir = True 
            # search all jsonl file in the data path
            self.anno_files = [ file for file in  os.listdir(self.data_path) if '.jsonl' in file] 
            
            self.anno_files.sort()
            self.anno_files = [ os.path.join(self.data_path, file) for file in self.anno_files]

        else:
            self.data_path_is_dir = False

        self.image_path = image_path
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.dataset_type = dataset_type
        self.min_active_tokens_ratio = min_active_tokens_ratio
        self.ds_name = ds_name
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        assert 'plain' in self.template_name, 'Only plain template is supported for pretraining with packed data.'
        assert not self.is_train, "Data augmentation is unnecessary for pretraining with packed data."

        self.tokenizer = tokenizer
        self.tcs_loader = tcs_loader
        self.transform = None

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        self.group_by_length = group_by_length
        assert not self.group_by_length, "Group_by_length is unnecessary for pretraining with packed data."

        self.dynamic_image_size = dynamic_image_size
        self.dynamic_image_version = dynamic_image_version
        if self.dynamic_image_version is None:
            self.dynamic_image_version = "v0"
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.max_dynamic_images = max_dynamic_images
        self.max_multi_image_dynamic_patch = max_multi_image_dynamic_patch

        # hyperparameters for interleaved documents
        self.max_num_images = max_num_images
        self.max_tokens = tokenizer.model_max_length
        assert self.max_num_images > 0

        # hyperparameters for data augmentation
        self.image_switch_prob = image_switch_prob

        if get_rank() == 0:
            logger.info(
                f"{self.ds_name=}, {self.image_switch_prob=}, {self.image_path=}, {self.data_path=}"
            )

        self.use_img_start_end_token = use_img_start_end_token
        if self.use_img_start_end_token:
            assert self.max_tokens >= self.num_image_token + 2
            self.image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token}{IMG_END_TOKEN}"
        else:
            assert self.max_tokens >= self.num_image_token
            self.image_tokens = f"{IMG_CONTEXT_TOKEN * self.num_image_token}"

        self.type_id = type_id
        # lazy init
        self.worker_id = None
        self.worker_state_key = None

        self.rng = np.random.default_rng(seed=42)

        if self.data_path_is_dir:
            self.start_file_idx = 0
            self.end_file_idx = len(self.anno_files)
        else:
            self.start_file_idx = meta.get('start_file_idx', 0)
            self.end_file_idx = meta['shard_end_id']
        self._state_dict = {
            'file_shift': 0,
            'bytes_offset': 0,
            'line_shift': 0,
            'success_cnt': 0,
            'fail_cnt': 0,
        }
        
        # parameters for native resolution
        self.patch_size = gpc.config.data.get('patch_size', 16)
        self.downsample_ratio = gpc.config.data.get('down_sample_ratio', 0.5)
        self.max_pixels = gpc.config.data.get('max_pixels', 4096*4096)
        self.min_pixels = gpc.config.data.get('min_pixels', 256*256)

    def load_state_dict(self, state_dict):
        self._state_dict.update(state_dict)

    def reset(self, ):
        # reset
        self._state_dict = {
            'file_shift': 0,
            'bytes_offset': 0,
            'line_shift': 0, # 用于整体的文件行数计数
            'success_cnt': 0,
            'fail_cnt': 0,
        }

    def load_image(self, image_path_or_url):
        try:
            return Image.open(image_path_or_url).convert("RGB")
        except Exception as e:
            raise Exception(f"Failed to load image {image_path_or_url}, exception info: {e}")


    def _get_mmap(self, data_path):
        try:
            f = open(data_path, "rb")  # pylint: disable=consider-using-with
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.handles = [f, mm]
            return self.handles[-1]
        except Exception as e:
            logger.error(f"Failed to mmap file {data_path}, exception info: {e}")
        
        return None
    
    def get_sample(self, line):
        try:
            sample = json.loads(line)
            tokenized_sample = self.tokenize_sample(sample)
            splitted_sample = self.split_sample(tokenized_sample)
            return splitted_sample
        except Exception as e:
            logger.error(f"[{self.ds_name}] [Worker id {self.worker_id}] Encountered error with {type(e)}& Error info: {e}")
            return None
        

    def split_sample(self, sample):
        return [sample]

    def __iter__(self):

        if self.transform is None:
            if getattr(self, "dynamic_image_version", False):
                assert not self.pad2square, "pad2square is not supported for native resolution"
                self.transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, resize=False)
            else:
                self.transform = build_transform(
                    is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square
                )

        local_worker_id = 0 if get_worker_info() is None else get_worker_info().id
        local_num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = local_num_workers * self.data_rank + local_worker_id
        num_workers = local_num_workers * self.data_world_size
        shard_worker_id = worker_id
        if num_workers > 1:
            # Avoid over-sharding small datasets into mostly-empty worker partitions.
            min_samples_per_worker = max(
                int(gpc.config.data.get("worker_distributed_min_samples_per_worker", 16)),
                1,
            )
            num_workers = min(
                num_workers,
                max(1, math.ceil(self.length / min_samples_per_worker)),
            )
            shard_worker_id = worker_id % num_workers

        self.worker_id = worker_id
        self.num_workers = num_workers

        self.worker_state_key = f"work_state_{self.worker_id}"
        
        repeat_time = math.ceil(self.repeat_time) # 文件需要打开的次数
        # if self._state_dict['file_shift'] >= self.end_file_idx:
        if self._state_dict['file_shift'] >= repeat_time*self.end_file_idx:
            # the dataset is exhausted
            self.reset()
            raise StopIteration
        
        file_shift = self._state_dict['file_shift']
        # for file_idx in range(self.start_file_idx + file_shift, self.end_file_idx):
        for file_idx in range(self.start_file_idx + file_shift, repeat_time*self.end_file_idx):
            if self.data_path_is_dir:
                current_shard_name = self.anno_files[file_idx % len(self.anno_files)]
            else:
                current_shard_name = os.path.join(self.data_path.format(i=file_idx))
            
            file: TextIOWrapper = self._get_mmap(current_shard_name)
            if file is None:
                logger.error(f"[{worker_id=}] [{self.ds_name}] Skip file {current_shard_name}")
                self._state_dict['file_shift'] += 1
                self._state_dict['bytes_offset'] = 0
                continue
            
            file.seek(0, 0)
            relative_offset = self._state_dict["bytes_offset"] - file.tell()
            assert relative_offset >= 0, f"Invalid offset {relative_offset} in file {current_shard_name}"
            file.seek(relative_offset, 1)

            line_offset = -1 #用于文件内部计数
            for line_offset, line in enumerate(iter(file.readline, b"")):
                old_bytes_offset = self._state_dict["bytes_offset"]
                old_line_shift = self._state_dict["line_shift"]

                self._state_dict["bytes_offset"] = file.tell()
                self._state_dict['line_shift'] += 1
                file.madvise(mmap.MADV_DONTNEED, 0, file.tell())

                if old_line_shift % num_workers == worker_id:
        

                    # if next_sample is not None:
                    if self.get_sample(line):
                        self._state_dict['success_cnt'] += 1
                        yield line

                    else:
                        self._state_dict['fail_cnt'] += 1

                        s = self._state_dict['success_cnt']
                        f = self._state_dict['fail_cnt']
                        if s/(s+f+1e-6) < 0.5 and (s+f) > 10:
                            logger.warning(
                                f"[{worker_id=}] [{self.ds_name}] success_rate: {s/(s+f+1e-6)*100:.2f}%, "
                            )
        
            if gpc.is_rank_for_log():
                logger.info(f"[{worker_id}] dataset {self.ds_name} {file_idx}-th datafile {current_shard_name} has been used up & the end line is {self._state_dict['line_shift']}")

            self._state_dict['file_shift'] += 1
            self._state_dict['bytes_offset'] = 0


        logger.info(f"[{worker_id}] dataset {self.ds_name} has been ran out of!!!")

        self.reset()
        raise StopIteration
            

class ImageTextPairDataset(BaseDataset):
    """ImageTextPairDataset"""

    def tokenize_sample(self, sample):
        text = sample[self.caption_key]

        image = sample["image"]
        image = os.path.join(self.image_path, image)
        image = self.load_image(image)
        pixel_values = [self.transform(image)]
        pixel_values = torch.stack(pixel_values)

        num_patches = pixel_values.size(0)

        # preprocess and tokenize text
        if self.image_switch_prob > 0 and self.image_switch_prob > self.rng.random():
            text = f"{text}{self.tokenizer.eos_token}\n{self.image_tokens}"
        else:
            text = f"{self.image_tokens}\n{text}{self.tokenizer.eos_token}"

        input_ids = self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).input_ids

        num_image_token = (input_ids == self.img_token_id).sum().item()
        if num_image_token != self.num_image_token * num_patches:
            raise RuntimeError(f"Find {num_image_token=}, while {self.num_image_token * num_patches=}")

        num_image_start_token = (input_ids == self.img_start_token_id).sum().item()
        if num_image_start_token != num_patches:
            raise RuntimeError(f"Find {num_image_start_token=}, while {num_patches=}")

        num_image_end_token = (input_ids == self.img_end_token_id).sum().item()
        if num_image_end_token != num_patches:
            raise RuntimeError(f"Find {num_image_end_token=}, while {num_patches=}")

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        labels = torch.where(labels == self.img_start_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_end_token_id, IGNORE_TOKEN_ID, labels)
        if self.tokenizer.bos_token_id is not None:
            labels = torch.where(labels == self.tokenizer.bos_token_id, IGNORE_TOKEN_ID, labels)

        # ignore </s> directly following the last image
        if input_ids[0][-2] == self.img_end_token_id:
            assert labels[0][-1] == self.tokenizer.eos_token_id
            labels[0][-1] = IGNORE_TOKEN_ID

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret


class InterleavedDataset(BaseDataset):
    """InterleavedDataset"""

    def get_img_filename(self, web_url, imgmeta):
        
        if not self.img_path_in_meta:
            return web_url

        if 'filename' in imgmeta:
            return imgmeta['filename']

        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig


    def parse_sample(self, sample):
        images = sample["images"]
        texts = sample["texts"]
        metadata = sample["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        valid_image = sample.get("valid_image", [True] * sum(img is not None for img in images))

        assert len(images) == len(texts), "len(images) != len(texts)"

        for seg_idx, (img, txt) in enumerate(zip(images, texts)):
            if txt == "<image>":
                texts[seg_idx] = None
            assert (img is None) != (txt in ["<image>", None]), f"img {img} txt {txt} {sample}"

        num_images = sum(img is not None for img in images)
        num_placeholders = sum(txt is None for txt in texts)
        assert (
            num_images == num_placeholders == len(valid_image)
        ), f"{num_images=}, {num_placeholders=}, {len(valid_image)=}, {sample=}"

        for _txt, _img, _imgmeta in zip(texts, images, metadata):
            if self.img_path_in_meta:
                assert( _img is None) == (_imgmeta is None), "( _img is None) == (_imgmeta is None)"
            else:
                if _img is None:
                    assert _imgmeta is None, "( _img is None) when (_imgmeta is None)"

        return images, texts, metadata, valid_image

    def tokenize_sample(self, sample):
        # parse sample and check
        images, texts, metadata, valid_image = self.parse_sample(sample)
        # get valid images
        images = [
            os.path.join(self.image_path, self.get_img_filename(img, imgmeta))
            for img, imgmeta in zip(images, metadata)
            if img is not None
        ]

        num_images = sum(valid_image)
        if num_images == 0:
            raise Exception("current sample do not contain any image.")

        # load images
        images_to_load = []
        image_load_fail_cnt = 0
        for img_idx, (img, valid) in enumerate(zip(images, valid_image)):
            if not valid:
                continue

            try:
                images_to_load.append(self.load_image(img))
            except Exception as e:
                image_load_fail_cnt += 1
                valid_image[img_idx] = False
                if self.worker_id == 0:
                    logger.info(f"fail to load image: {img}, info: {e}")

            if image_load_fail_cnt >= max(num_images // 2, 1):
                raise Exception(f"failed to load {image_load_fail_cnt} images from {num_images} images")

        images_to_load_patches, num_tiles = [], []
        num_image = len(images_to_load)
        # if self.dynamic_image_size and num_image <= self.max_dynamic_images:
        if self.dynamic_image_size:
            
            for image in images_to_load:
                if self.dynamic_image_version == "native_resolution":
                    patch = dynamic_preprocess_native_resolution(image,
                                                                 min_pixels=self.min_pixels, 
                                                                 max_pixels=self.max_pixels if num_image == 1 else max(self.max_pixels * 2 // num_image, self.min_pixels), 
                                                                 size_factor=int(self.patch_size / self.downsample_ratio))
                    images_to_load_patches.append(patch)
                    w, h = patch.size
                    num_tiles.append(int(w * h // self.patch_size**2 * self.downsample_ratio**2))
                else:
                    raise NotImplementedError(f"dynamic_image_version must be 'native_resolution', got {self.dynamic_image_version!r}")
                    num_tiles.append(len(patches))
            
        # preprocess images
        if self.dynamic_image_size and self.dynamic_image_version == "native_resolution":
            pixel_values = [self.transform(image) for image in images_to_load_patches]
        else:
            if self.dynamic_image_size:
                pixel_values = [self.transform(image) for image in images_to_load_patches]
            else:
                pixel_values = [self.transform(image) for image in images_to_load]
            # raise Error when no images in this document
            pixel_values = torch.stack(pixel_values)

        num_patches = len(pixel_values)

         # if self.dynamic_image_size and num_image <= self.max_dynamic_images:
        if self.dynamic_image_size and self.dynamic_image_version != "native_resolution":
            assert sum(num_tiles) == pixel_values.size(0), f"{sum(num_tiles)=}, {pixel_values.size(0)=}"

        # preprocess and tokenize text
        image_idx = 0
        for i in range(len(texts)):
            if texts[i] is None:
                if valid_image[image_idx]:
                    texts[i] = "<image>"
                image_idx += 1
        texts = [_ for _ in texts if _]

        if self.image_switch_prob > 0:
            for text_idx in range(1, len(texts)):
                if (
                    texts[text_idx - 1] == "<image>"
                    and texts[text_idx] != "<image>"
                    and self.image_switch_prob > self.rng.random()
                ):
                    texts[text_idx - 1] = texts[text_idx]
                    texts[text_idx] = "<image>"

        text = " ".join(texts)
        text = f"{text}{self.tokenizer.eos_token}"
        text = text.replace("<image> ", "<image>\n").replace(" <image>", "\n<image>")
        # if self.dynamic_image_size and num_image <= self.max_dynamic_images:
        if self.dynamic_image_size:
            image_count = text.count("<image>")
            if image_count != len(num_tiles):
                raise RuntimeError(f"image_count error {image_count=}, while {len(num_tiles)=}")
            else:
                text_after_replace = text
                for count in range(image_count):
                    if self.use_img_start_end_token:
                        if self.dynamic_image_version == 'native_resolution':
                            image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_tiles[count]}{IMG_END_TOKEN}"
                        else:
                            image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token*num_tiles[count]}{IMG_END_TOKEN}"
                        text_after_replace = text_after_replace.replace("<image>", image_tokens, 1)
                    else:
                        text_after_replace = text_after_replace.replace("<image>", self.image_tokens*num_tiles[count], 1)
        else:
            text_after_replace = text.replace("<image>", self.image_tokens)

        input_ids = self.tokenizer(
            text_after_replace,
            max_length=self.max_tokens,
            truncation=False,
            padding=False,
            return_tensors="pt",
        ).input_ids

        num_image_token = (input_ids == self.img_token_id).sum().item()
        if getattr(self, "dynamic_image_version", None) == "native_resolution":
            num_image_token_expected = sum(num_tiles)
        else:
            num_image_token_expected = self.num_image_token * num_patches
        if num_image_token != num_image_token_expected:
            raise RuntimeError(f"Find {num_image_token=}, while expecting {num_image_token_expected=}")

        num_image_start_token = (input_ids == self.img_start_token_id).sum().item()
        if num_image_start_token != num_patches:
            raise RuntimeError(f"Find {num_image_start_token=}, while {num_patches=}")

        num_image_end_token = (input_ids == self.img_end_token_id).sum().item()
        if num_image_end_token != num_patches:
            raise RuntimeError(f"Find {num_image_end_token=}, while {num_patches=}")

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        labels = torch.where(labels == self.img_start_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_end_token_id, IGNORE_TOKEN_ID, labels)
        if self.tokenizer.bos_token_id is not None:
            labels = torch.where(labels == self.tokenizer.bos_token_id, IGNORE_TOKEN_ID, labels)

        # ignore </s> directly following the last image
        if input_ids[0][-2] == self.img_end_token_id:
            assert labels[0][-1] == self.tokenizer.eos_token_id
            labels[0][-1] = IGNORE_TOKEN_ID

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        if self.dynamic_image_version == 'native_resolution':
            assert isinstance(pixel_values, list), f"pixel_values should be a list, but got {type(pixel_values)}"

        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def split_sample(self, sample):
        if self.split_version == 'v0':
            # weiyun implements it for vit-moe
            return self.split_sample_v0(sample)
        elif self.split_version == 'v1':
            return self.split_sample_v1(sample)
        else:
            raise NotImplementedError(f"find unsupported split_version {self.split_version}")
    

    def split_sample_v0(self, sample):
        if len(sample['pixel_values']) <= self.max_num_images and sample['input_ids'].size(0) <= self.max_tokens:
            return [sample]

        splitted_samples = []
        img_start_idx_list = (sample["input_ids"] == self.img_start_token_id).nonzero().squeeze(1).tolist()
        img_end_idx_list = (sample["input_ids"] == self.img_end_token_id).nonzero().squeeze(1).tolist()

        assert len(img_start_idx_list) == len(img_end_idx_list), "len(img_start_idx_list) == len(img_end_idx_list)"

        def _truncate(sample_to_truncte, start_idx, end_idx, start_img_idx, end_img_idx):
            new_sample = {}
            for k in sample_to_truncte:
                if k in ["input_ids", "labels", "attention_mask", "position_ids"]:
                    new_sample[k] = sample_to_truncte[k][start_idx:end_idx]
                elif k in ["pixel_values", "image_flags"]:
                    new_sample[k] = sample_to_truncte[k][start_img_idx:end_img_idx]
                else:
                    raise NotImplementedError(f"find unsupported keys: {k} from {sample_to_truncte.keys()}")
            return new_sample
        # split之后所有的subsample都是image tokens开头
        i = 0
        for i in range(self.max_num_images, len(img_start_idx_list), self.max_num_images):
            curr_img_start_idx = img_start_idx_list[i]
            prev_img_end_idx = img_start_idx_list[i - self.max_num_images]

            if i - self.max_num_images == 0:
                prev_img_end_idx = 0

            truncated_sample = _truncate(
                sample,
                start_idx=prev_img_end_idx,
                end_idx=curr_img_start_idx,
                start_img_idx=i - self.max_num_images,
                end_img_idx=i,
            )
            if PackedDataset.check_valid(truncated_sample, self.min_active_tokens_ratio):
                splitted_samples.append(truncated_sample)

        if i < len(img_start_idx_list):
            truncated_sample = _truncate(
                sample,
                start_idx=img_start_idx_list[i],
                end_idx=len(sample["input_ids"]),
                start_img_idx=i,
                end_img_idx=len(sample["pixel_values"]),
            )
            if PackedDataset.check_valid(truncated_sample, self.min_active_tokens_ratio):
                splitted_samples.append(truncated_sample)

        final_splitted_samples = []
        for each_sample in splitted_samples:
            final_splitted_samples.extend(
                PackedDataset.split_buffer(
                    buffer=each_sample,
                    max_tokens=self.max_tokens,
                    img_start_token_id=self.img_start_token_id,
                    img_token_id=self.img_token_id,
                    img_end_token_id=self.img_end_token_id,
                )
            )

        return final_splitted_samples
    
    def split_sample_v1(self, sample):
        if len(sample['pixel_values']) <= self.max_num_images and sample['input_ids'].size(0) <= self.max_tokens:
            return [sample]


        def _image_is_splitted(input_ids, cut_idx):
            is_image_start = input_ids[cut_idx].item() == self.img_start_token_id
            is_image_token = input_ids[cut_idx].item() == self.img_token_id
            is_image_end = input_ids[cut_idx].item() == self.img_end_token_id
            return is_image_start or is_image_token or is_image_end

        def _split(sample_to_split, cut_idx, cut_img_idx, drop_right):
           
            left_sample = {}
            right_sample = {} if not drop_right else None
            for k in sample_to_split:
                if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index', 'type_ids']:
                    left_sample[k] = sample_to_split[k][:cut_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][cut_idx:]
                elif k in ['pixel_values', 'image_flags']:
                    left_sample[k] = sample_to_split[k][:cut_img_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][cut_img_idx:]
                else:
                    raise NotImplementedError(f"find unsupported keys: {k} from {sample_to_split.keys()}")
            return left_sample, right_sample
        

        # 截断之后的字符同时满足两个条件：
        # 1. 图像数量不超过max_num_images
        # 2. token数量不超过max_tokens
        # 3. 不把图像tokens拆开

        splitted_sample = []
        while sample['input_ids'].size(0) > self.max_tokens or len(sample['pixel_values']) > self.max_num_images:

            
            img_start_idx_list = (sample['input_ids'] == self.img_start_token_id).nonzero().squeeze(1).tolist()
            img_end_idx_list = (sample['input_ids'] == self.img_end_token_id).nonzero().squeeze(1).tolist()

            assert len(img_start_idx_list) == len(img_end_idx_list), "len(img_start_idx_list) != len(img_end_idx_list) when split sample"

            # 从图片的角度思考截断位置
            cut_idx_for_img = img_start_idx_list[self.max_num_images] if len(img_start_idx_list) > self.max_num_images else len(sample['input_ids'])
            # 从文本的角度思考截断位置
            cut_idx_for_text = self.max_tokens if len(sample['input_ids']) > self.max_tokens else len(sample['input_ids'])

            cut_length = min(cut_idx_for_img, cut_idx_for_text)

            cut_img_idx = bisect.bisect_left(img_start_idx_list, cut_length)
            # 考虑是否截断了图片token
            if _image_is_splitted(sample['input_ids'], cut_length):
                if sample['input_ids'][cut_length] == self.img_start_token_id:
                    # 截断位置是图片的开头 无事发生 照常截断就行
                    assert cut_length == img_start_idx_list[cut_img_idx]
                    
                else:
                    # 截断位置是图片的中间 从上一个图片的开头开始截断
                    cut_length = img_start_idx_list[cut_img_idx - 1]
                    cut_img_idx = cut_img_idx - 1
            
            # drop 掉最后面没有图片的末尾文本
            if cut_img_idx == len(img_start_idx_list):
                drop_right = True
            else:
                drop_right = False    

            left, right = _split(sample, cut_length, cut_img_idx, drop_right=drop_right)

            assert (left['input_ids'] == self.img_end_token_id).sum() == (left['input_ids'] == self.img_start_token_id).sum() == len(left['pixel_values'])
            if right is not None:
                assert (right['input_ids'] == self.img_end_token_id).sum() == (right['input_ids'] == self.img_start_token_id).sum() == len(right['pixel_values'])

            if len(left['pixel_values']) >= 1 and PackedDataset.check_valid(left, self.min_active_tokens_ratio):
                splitted_sample.append(left)

            if right is None or len(right['pixel_values']) == 0:
                sample = None
                break

            sample = right

        if sample is not None and PackedDataset.check_valid(sample, self.min_active_tokens_ratio):
            splitted_sample.append(sample)

        logger.debug(
            f"split a sample into {len(splitted_sample)} samples, "
            f"current max_tokens={self.max_tokens}"
        )

        # check sample validity
        for each_sample in splitted_sample:
            assert each_sample['input_ids'].size(0) <= self.max_tokens, f"{each_sample['input_ids'].size(0)=} {self.max_tokens=}"
            assert len(each_sample['pixel_values']) <= self.max_num_images, f"{len(each_sample['pixel_values'])=} {self.max_num_images=}"

        return splitted_sample
           

# NOTE:
class PackedDataset(IterableDataset):
    """PackedDataset"""

    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List[Union[ImageTextPairDataset, InterleavedDataset]],
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 32768,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        nlp_mm_sampling_fixed_token: bool = False,
        packed_buffer_stale_threshold: int = 200,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        if gpc.config.data.get("dynamic_image_version", "v3") == "native_resolution":
            # TODO: fix hard code
            self.num_images_expected = 32768
            assert self.strict_mode == False, "strict_mode should be False when dynamic_image_version is native_resolution"
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data
        self.packed_buffer_stale_threshold = max(int(packed_buffer_stale_threshold), 1)

        self.max_packed_tokens = max_packed_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        self.dataset_type = [d.dataset_type for d in self.datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]

        self.datasets = list(self.datasets_orig)
        self.dataset_weight = list(self.dataset_weight_orig)

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            "sample_info": {d.ds_name: 0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f"deduplicated ds_name: {ds_name_list}"

        for ds in self.datasets:
            if hasattr(ds, "max_num_images"):
                if ds.max_num_images > self.num_images_expected:
                    logger.warning(f"{ds.max_num_images=} of {ds.ds_name} is larger than {self.num_images_expected=}")
                    ds.max_num_images = num_images_expected

            if hasattr(ds, 'max_tokens') and ds.max_tokens > self.max_packed_tokens:
                logger.warning(f"{ds.max_tokens=} of {ds.ds_name} is larger than {self.max_packed_tokens=}")
                ds.max_tokens = self.max_packed_tokens

            self._state_dict[ds.ds_name] = {}
        
        self.nlp_mm_sampling_fixed_token = nlp_mm_sampling_fixed_token
        
        self.resume_state_dict = {}

        if get_rank() == 0:
            logger.info(
                f"Loaded dataset to pack: {ds_name_list}, "
                f"{self.num_images_expected=}, {self.max_packed_tokens=}, "
                f"{self.replacement=}, {self.allow_overflow=}",
            )
            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f"{ds.ds_name:<25}: {ds_w*100:.4f}%")
            temp = "\n".join(temp)
            logger.info(f"Sampling prob for each dataset:\n{temp}")

        if self.allow_empty_data:
            logger.warning("allow_empty_data is enabled, note that empty data may be generated!")

    def load_state_dict(self, state_dict, custom_infos=None):


        self.resume_state_dict.update(state_dict)
        

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        try:
            if isinstance(self.datasets[current_dataset_idx], sensenovavl.data.dataset_interleaved_iterable.BaseDataset):
                line = next(self.dataset_iter_list[current_dataset_idx])
                current_sample = self.datasets[current_dataset_idx].get_sample(line)[0]
                current_sample["type_ids"] = (
                    torch.zeros_like(current_sample["input_ids"]) + self.datasets[current_dataset_idx].type_id
                )
                current_sample["meta_info"] = copy.deepcopy(self.datasets[current_dataset_idx]._state_dict)
            elif isinstance(self.datasets[current_dataset_idx], sensenovavl.data.multimodal_dataset.LazySupervisedDataset):
                line = next(self.dataset_iter_list[current_dataset_idx])
                current_sample = self.datasets[current_dataset_idx].get_sample(line)
                current_sample["meta_info"] = copy.deepcopy(self.datasets[current_dataset_idx]._state_dict)
            else:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
        except Exception as e:
            logger.error(f"Dataloader caught exception type  {type(e)} which is: {e}")
            # except StopIteration:/
            if self.replacement:
                logger.info(
                    f"[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} "
                    "is exhausted, restart it."
                )

                try:
                    self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                    if isinstance(
                        self.datasets[current_dataset_idx], sensenovavl.data.dataset_interleaved_iterable.BaseDataset
                    ):
                        line = next(self.dataset_iter_list[current_dataset_idx])
                        current_sample = self.datasets[current_dataset_idx].get_sample(line)[0]
                        current_sample["type_ids"] = (
                            torch.zeros_like(current_sample["input_ids"]) + self.datasets[current_dataset_idx].type_id
                        )
                        current_sample["meta_info"] = copy.deepcopy(self.datasets[current_dataset_idx]._state_dict)
                    elif isinstance(
                        self.datasets[current_dataset_idx], sensenovavl.data.multimodal_dataset.LazySupervisedDataset
                    ):
                        line = next(self.dataset_iter_list[current_dataset_idx])
                        current_sample = self.datasets[current_dataset_idx].get_sample(line)
                        current_sample["meta_info"] = copy.deepcopy(self.datasets[current_dataset_idx]._state_dict)
                    else:
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                except Exception as e:
                    logger.error(f"{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name} with error {type(e)} {e}!")
                    self.dataset_weight[current_dataset_idx] = 0


                    # if len(self.datasets) == 0:
                    if all(weight == 0 for weight in self.dataset_weight):
                        logger.info(f"[Worker id {self.worker_id}] All Datasets are exhausted, restart them.")
                        self.dataset_weight = [w for w in self.dataset_weight_orig]
                        self.dataset_iter_list = [iter(d) for d in self.datasets]

                    return None
            else:
                logger.error(f"{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}!")
                self.dataset_weight[current_dataset_idx] = 0.0


                # if len(self.datasets) == 0:
                if all(weight == 0 for weight in self.dataset_weight):
                    logger.info(f"[Worker id {self.worker_id}] All Datasets are exhausted, restart them.")
                    self.dataset_weight = [w for w in self.dataset_weight_orig]
                    self.dataset_iter_list = [iter(d) for d in self.datasets]

                return None # self.next_data(np.random.choice(len(self.datasets)))

        current_ds_name = self.datasets[current_dataset_idx].ds_name
        

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}
        
        if self.datasets[current_dataset_idx].ds_name == 'llm_mixed_training':
            
            ds_state_dict = current_sample.pop('state_dict', None)
            
            if ds_state_dict is not None:
                for key in ['rng_state', 'consumed_samples', "used_epochs", "epochs_to_use"]:
                # rng_state will be discard
                # for key in [ 'consumed_samples', "used_epochs", "epochs_to_use"]:
                    self._state_dict[current_ds_name][self.worker_state_key][key] = ds_state_dict[key]
                
                ds_state_dict['dataset_consumed_tokens'] = {
                    ds_state_dict['dataset_name']: len(current_sample['data']['input_ids'])
                }
                for key in ["dataset_consumed_tokens", "multiple_packed_states"]:
                    if key not in self._state_dict[current_ds_name][self.worker_state_key]:
                        self._state_dict[current_ds_name][self.worker_state_key][key] = {}
                    self._state_dict[current_ds_name][self.worker_state_key][key].update(ds_state_dict[key])     
            self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1

            current_sample = current_sample.pop('data')
            # already tokenized and packed
            current_sample['already_packed'] = True

        else:
            assert 'type_ids' in current_sample, 'should have type ids'
            meta_info = current_sample.pop('meta_info', {})
            # 只保留 resume 需要的状态，避免不必要的数据传播
            needed_keys = {'file_shift', 'bytes_offset', 'line_shift', 'success_cnt', 'fail_cnt'}
            filtered_meta = {k: v for k, v in meta_info.items() if k in needed_keys}
            self._state_dict[current_ds_name][self.worker_state_key].update(**filtered_meta)
            self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster

        find = False
        find_idx = -1
        best_remaining = self.max_packed_tokens + 1  # track tightest fit
        overflow_idx = -1  # fallback for allow_overflow

        num_tokens_current = new_sample["input_ids"].size(0)
        num_images_current = 0 if new_sample["pixel_values"] is None else len(new_sample["pixel_values"])
        for buffer_idx, buffer in enumerate(buffer_list):
            num_images_buffer = 0 if buffer["pixel_values"] is None else len(buffer["pixel_values"])
            if num_images_buffer + num_images_current > self.num_images_expected:
                continue

            num_merged_tokens = num_tokens_current + buffer["input_ids"].size(0)

            if num_merged_tokens <= self.max_packed_tokens:
                remaining = self.max_packed_tokens - num_merged_tokens
                if remaining < best_remaining:
                    best_remaining = remaining
                    find = True
                    find_idx = buffer_idx
                    if remaining == 0:
                        break  # perfect fit, no need to search further
            elif self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                if overflow_idx == -1:
                    overflow_idx = buffer_idx

        if not find and overflow_idx >= 0:
            find = True
            find_idx = overflow_idx

        if find:
            return buffer_list.pop(find_idx)
        return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample["data_index"] = torch.zeros_like(new_sample["input_ids"])
            return new_sample

        new_sample["data_index"] = torch.ones_like(new_sample["input_ids"]) + buffer["data_index"][-1].item()

        for k in buffer:
            if k not in new_sample:
                continue
            if new_sample[k] is not None:
                if isinstance(buffer[k], str):
                    assert buffer[k] == new_sample[k], f"{buffer[k]} != {new_sample[k]}"
                    buffer[k] = buffer[k]
                elif buffer[k] is not None:
                    if isinstance(buffer[k], list):
                        buffer[k].extend(new_sample[k])
                    else:
                        try:
                            buffer[k] = torch.cat([buffer[k], new_sample[k]])
                        except:
                            print(f"Failed to concatenate {k} with shapes {buffer[k].shape} and list")

                else:
                    buffer[k] = new_sample[k]
            else:
                buffer[k] = buffer[k]

        return buffer

    @staticmethod
    def check_valid(sample_to_check, min_active_tokens_ratio=1/256):
        num_ignore_tokens = (sample_to_check['labels'] == IGNORE_TOKEN_ID).sum()
        num_tokens = sample_to_check['labels'].numel()
        if num_tokens == 0:
            return False 
        return (1 - num_ignore_tokens / num_tokens) > min_active_tokens_ratio

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id):
        if buffer["input_ids"].size(0) <= max_tokens:
            return [buffer]

        def _image_is_splitted(input_ids, cut_idx):
            is_image_start = input_ids[cut_idx].item() == img_start_token_id
            is_image_token = input_ids[cut_idx].item() == img_token_id
            is_image_end = input_ids[cut_idx].item() == img_end_token_id
            return is_image_start or is_image_token or is_image_end

        def _split(sample_to_split, left_idx, right_idx, left_img_idx, right_img_idx):
            assert (right_idx is None) == (right_img_idx is None)

            left_sample = {}
            right_sample = {} if right_idx is not None else None
            for k in sample_to_split:
                if k in ["input_ids", "labels", "attention_mask", "position_ids", "data_index", "type_ids"]:
                    left_sample[k] = sample_to_split[k][:left_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_idx:]
                elif k in [
                    "pixel_values",
                    "image_seq_lens",
                    "image_flags",
                    "image_for_gen_flags",
                    "image_for_gen_loss_flags",
                    "is_image_duplicated_for_und_flags",
                ]:
                    left_sample[k] = sample_to_split[k][:left_img_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_img_idx:]
                else:
                    raise NotImplementedError(f"find unsupported keys: {k} from {sample_to_split.keys()}")
            return left_sample, right_sample

        # NOTE: 给定 <img_1> <text_1_1> <text_1_2> <img_2> <text_2_1> <text_2_2>
        # 情况1: 从文本部分的中间切开，此时右侧部分会对齐到下一张图片的开头
        #       <img_1> <text_1_1> 和 <img_2> <text_2_1> <text_2_2>
        # 情况2: 从图像的中间切开，此时切口会对齐到这张图像的<img>开始标识

        # split完成后：左侧数据达到最大长度，但是图像数量不足，如果明显少，就丢掉
        # 右侧部分留着去buffer

        # 补充1: 原始情况下必然有图
        # 补充2: 右侧的部分没有图像时，直接丢弃，因为他们原本condition on图像
        # 补充3: 左侧部分没有图像的情况只会在第一个iter出现，原本就uncondition，但是vit会没有梯度，所以还是丢掉

        splitted_buffer = []
        while buffer["input_ids"].size(0) > max_tokens:
            img_start_idx_list = (buffer["input_ids"] == img_start_token_id).nonzero().squeeze(1).tolist()
            img_end_idx_list = (buffer["input_ids"] == img_end_token_id).nonzero().squeeze(1).tolist()
            assert len(img_start_idx_list) == len(img_end_idx_list)

            if _image_is_splitted(buffer["input_ids"], max_tokens):
                cut_idx = bisect.bisect_left(img_start_idx_list, max_tokens)
                if buffer["input_ids"][max_tokens] == img_start_token_id:
                    assert max_tokens == img_start_idx_list[cut_idx]
                    cut_left_idx = img_start_idx_list[cut_idx]
                    cut_left_img_idx = cut_idx
                else:
                    cut_left_idx = img_start_idx_list[cut_idx - 1]
                    cut_left_img_idx = cut_idx - 1
                cut_right_idx = cut_left_idx
                cut_right_img_idx = cut_left_img_idx
            else:
                cut_img_idx = bisect.bisect(img_start_idx_list, max_tokens)
                if cut_img_idx < len(img_start_idx_list):
                    cut_right_idx = img_start_idx_list[cut_img_idx]
                    cut_right_img_idx = cut_img_idx
                else:
                    # drop last segment without images
                    cut_right_idx = None
                    cut_right_img_idx = None

                cut_left_idx = max_tokens
                cut_left_img_idx = (
                    cut_right_img_idx if cut_right_img_idx is not None else len(buffer["pixel_values"])
                )

            left, right = _split(
                sample_to_split=buffer,
                left_idx=cut_left_idx,
                left_img_idx=cut_left_img_idx,
                right_idx=cut_right_idx,
                right_img_idx=cut_right_img_idx,
            )

            assert (
                (left["input_ids"] == img_end_token_id).sum()
                == (left["input_ids"] == img_start_token_id).sum()
                == len(left["pixel_values"])
            )
            if right is not None:
                assert (
                    (right["input_ids"] == img_end_token_id).sum()
                    == (right["input_ids"] == img_start_token_id).sum()
                    == len(right["pixel_values"])
                )

            if len(left["pixel_values"]) >= 1 and PackedDataset.check_valid(left):
                splitted_buffer.append(left)

            if right is None or len(right["pixel_values"]) == 0:
                break

            buffer = right
            if buffer["input_ids"].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                splitted_buffer.append(buffer)
                break

        logger.debug(f"split a sample into {len(splitted_buffer)} samples, " f"current max_tokens={max_tokens}")
        return splitted_buffer

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer, iter_idx=0):
        # NOTE: in-place operation

        splitted_buffer = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
        )

        for each_buffer in splitted_buffer:
            if (
                each_buffer["pixel_values"] is not None
                and len(each_buffer["pixel_values"]) > self.num_images_expected
            ):
                logger.error(
                    f"Find a sample with {len(each_buffer['pixel_values'])} images, "
                    f"which exceeds {self.num_images_expected}"
                )
                continue

            if each_buffer["input_ids"].size(0) >= self.max_packed_tokens:
                assert each_buffer["input_ids"].size(0) == self.max_packed_tokens
                buffer_max_len_list.append(each_buffer)
                continue

            # Stamp birth iter for stale buffer eviction (only if not already set)
            if "_birth_iter" not in each_buffer:
                each_buffer["_birth_iter"] = iter_idx

            # insert into buffer_list with descending order of number of images
            num_images_new = 0 if each_buffer["pixel_values"] is None else len(each_buffer["pixel_values"])
            find_idx = len(buffer_list)
            for buffer_idx in range(len(buffer_list)):
                num_images_buffer = (
                    0 if buffer_list[buffer_idx]["pixel_values"] is None else len(buffer_list[buffer_idx]["pixel_values"])
                )
                if num_images_buffer < num_images_new:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        return buffer_list, buffer_max_len_list
    
    # will not be enabled on native resolution mode
    def pad_buffer(self, buffer):
        if buffer["pixel_values"] is not None and buffer["pixel_values"].size(0) == self.num_images_expected:
            return buffer

        num_pad_images = self.num_images_expected - buffer["pixel_values"].size(0)
        pad_images = torch.stack([torch.zeros_like(buffer["pixel_values"][0]) for _ in range(num_pad_images)])
        pad_image_flags = torch.tensor([0] * num_pad_images, dtype=torch.long)

        buffer["pixel_values"] = torch.cat([buffer["pixel_values"], pad_images])
        buffer["image_flags"] = torch.cat([buffer["image_flags"], pad_image_flags])

        return buffer

    def postprocess_buffer(self, buffer, custom_infos=None):
        # Remove internal tracking fields before yielding to downstream
        buffer.pop("_birth_iter", None)
        buffer["worker_state_key"] = self.worker_state_key
        # Use pickle serialization instead of copy.deepcopy to avoid
        # expensive recursive copy of the large nested _state_dict every yield
        buf = io.BytesIO()
        pickle.dump(self._state_dict, buf)
        buffer["worker_state_dict"] = buf.getvalue()
        buffer.pop("custom_infos", None)
        if custom_infos is not None and False:
            # Strip pixel_values from buffer_list before serialization
            # to avoid deep-copying hundreds of MB of image tensors
            if "buffer_list" in custom_infos:
                stripped = []
                for b in custom_infos["buffer_list"]:
                    sb = {k: v for k, v in b.items() if k not in ("pixel_values", "_birth_iter")}
                    sb["pixel_values"] = None
                    stripped.append(sb)
                custom_infos = {**custom_infos, "buffer_list": stripped}
            buf2 = io.BytesIO()
            pickle.dump(custom_infos, buf2)
            buffer["custom_infos"] = {self.worker_state_key: buf2.getvalue()}
        return buffer

    def print_log(self, iter_idx, buffer_list):  # pylint: disable=W0613
        if iter_idx % self.log_freq != 0:
            return

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # restore rng state if provided
        yeild_success = True
        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]

            if "rng_state" in custom_infos:
                try:
                    rng.bit_generator.state = custom_infos["rng_state"]
                except Exception as e:
                    logger.warning(f"[{self.worker_state_key}] failed to restore rng_state: {e}")

            if "yeild_success" in custom_infos:
                yeild_success = bool(custom_infos["yeild_success"])

            # optional: restore dataset_weight snapshot
            if "dataset_weight" in custom_infos:
                try:
                    self.dataset_weight = list(custom_infos["dataset_weight"])
                except Exception as e:
                    logger.warning(f"[{self.worker_state_key}] failed to restore dataset_weight: {e}")

        def _make_custom_infos():
            return {
                "buffer_list": buffer_list,
                "rng_state": rng.bit_generator.state,
                "yeild_success": yeild_success,
                # optional:
                "dataset_weight": self.dataset_weight,
                "iter_idx": iter_idx,
            }

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f"work_state_{self.worker_id}"

        # resume subdataset 
        if self.resume_state_dict is not None and len(self.resume_state_dict)>0:
            logger.info(f"resume sub-dataset states")
            for ds in self.datasets:
                if ds.ds_name in self.resume_state_dict:
                    if self.worker_state_key in self.resume_state_dict[ds.ds_name]:
                        ds.load_state_dict(self.resume_state_dict[ds.ds_name][self.worker_state_key])
                        if self.worker_state_key not in self._state_dict[ds.ds_name]:
                            self._state_dict[ds.ds_name][self.worker_state_key] = {}
                        self._state_dict[ds.ds_name][self.worker_state_key].update(self.resume_state_dict[ds.ds_name][self.worker_state_key])
                    else:
                        logger.info(f'NOTE: [Worker{self.worker_state_key}] does not resume dataset {ds.ds_name}!')
                
            self.resume_state_dict = {}

        self.datasets = [d for d in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
                ds.worker_id = worker_id
                ds.worker_state_key = f"work_state_{self.worker_id}"
                ds.num_workers = num_workers
                if self._should_log() and worker_id == 0:
                    logger.info(f"set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}")

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if "buffer_list" in custom_infos and isinstance(custom_infos["buffer_list"], list):
                buffer_list = custom_infos["buffer_list"]
                if self._should_log() and worker_id == 0:
                    logger.info(f"[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}")
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(f"{self.__class__.__name__} Rank {self.data_rank} " f"Worker {worker_id} begin to load data")

        if self._should_log():
            logger.info(f"Begin to iter, {len(buffer_list)=}")

        yeild_success = True
        yield_count = 0
        sync_interval = getattr(self, 'resume_packed_buffer_sync_interval', 100)

        # P2: dynamic weight adjustment based on remaining samples
        dynamic_weight_interval = 100  # re-compute every N iterations
        dynamic_weight_counter = 0
        while True:
            # Periodically adjust dataset_weight by remaining sample ratio
            dynamic_weight_counter += 1
            if (dynamic_weight_counter - 1) % dynamic_weight_interval == 0:
                new_weights = []
                for ds_idx, ds in enumerate(self.datasets):
                    orig_w = self.dataset_weight_orig[ds_idx]
                    if orig_w == 0 or self.dataset_weight[ds_idx] == 0:
                        new_weights.append(0.0)
                        continue
                    consumed = self._state_dict['sample_info'].get(ds.ds_name, 0)
                    # estimate per-worker total: ds.length spread across all workers
                    per_worker_total = max(ds.length / num_workers, 1)
                    remaining_ratio = max(1.0 - consumed / per_worker_total, 0.0)
                    new_weights.append(orig_w * remaining_ratio)
                weight_sum = sum(new_weights)
                if weight_sum > 0:
                    self.dataset_weight = new_weights

            if self.nlp_mm_sampling_fixed_token and not yeild_success:
                # will not sample the first nlp dataset 
                sampling_weight =  [0.0] + self.dataset_weight[1:]
                sampling_weight = [w/sum(sampling_weight) for w in sampling_weight]
                current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=sampling_weight)
            else:
                self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]

                current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)
                yeild_success = False

            current_sample = self.next_data(current_dataset_idx)

            if current_sample is None:
                # a adataset is over!!
                continue

            if current_sample.get('already_packed', False):
                # llm pretraining dataset
                if yield_count % sync_interval == 0:
                    yield self.postprocess_buffer(current_sample,  _make_custom_infos())
                else:
                    yield self.postprocess_buffer(current_sample)
                yeild_success = True
                yield_count += 1
                iter_idx += 1
                continue

            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer, iter_idx=iter_idx)

            while len(buffer_max_len_list) > 0:


                if self.strict_mode and len(buffer_max_len_list[0]['pixel_values']) != self.num_images_expected:
                    if buffer_max_len_list[0]['pixel_values'] is None:
                        raise NotImplementedError(f'为none的时候的padding func还没做好')
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_max_len_list.pop(0)), _make_custom_infos())
                    else:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_max_len_list.pop(0)))
                else:
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(buffer_max_len_list.pop(0), _make_custom_infos())
                    else:
                        yield self.postprocess_buffer(buffer_max_len_list.pop(0))
                yeild_success = True
                yield_count += 1

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'] is not None and len(buffer_list[0]['pixel_values']) > self.num_images_expected:
                logger.error(
                    f"num images of a buffer ({len(buffer_list[0]['pixel_values'])}) "
                    f"is larger than num_images_expected({self.num_images_expected})"
                )
                buffer_list.pop(0)

            while (
                len(buffer_list) > 0
                and buffer_list[0]["pixel_values"] is not None
                and len(buffer_list[0]["pixel_values"]) == self.num_images_expected
            ):
                if self.debug_mode:
                    debug_data = self.postprocess_buffer(buffer_list.pop(0),  _make_custom_infos())
                    while True:
                        yield debug_data.copy()

                if yield_count % sync_interval == 0:
                    yield self.postprocess_buffer(buffer_list.pop(0), _make_custom_infos())
                else:
                    yield self.postprocess_buffer(buffer_list.pop(0))
                yeild_success = True
                yield_count += 1

            # Proactively yield buffers that are nearly full by token count
            nearly_full_threshold = int(self.max_packed_tokens * 0.96)
            while len(buffer_list) > 0:
                max_tok_idx = max(range(len(buffer_list)), key=lambda i: buffer_list[i]["input_ids"].size(0))
                if buffer_list[max_tok_idx]["input_ids"].size(0) >= nearly_full_threshold:
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(buffer_list.pop(max_tok_idx), {'buffer_list': buffer_list})
                    else:
                        yield self.postprocess_buffer(buffer_list.pop(max_tok_idx))
                    yeild_success = True
                    yield_count += 1
                else:
                    break

            # Force yield stale buffers that have been sitting for too long
            stale_threshold = self.packed_buffer_stale_threshold
            stale_indices = [i for i, b in enumerate(buffer_list) if iter_idx - b.get("_birth_iter", iter_idx) >= stale_threshold]
            # Yield stale buffers in descending index order to avoid index shifting issues
            if stale_indices:
                stale_indices.sort(key=lambda i: buffer_list[i]["input_ids"].size(0), reverse=True)
                # Pop from highest index first to keep lower indices valid
                pop_indices = sorted(stale_indices, reverse=True)
                popped_buffers = [buffer_list.pop(i) for i in pop_indices]
                # Yield in token-count descending order (most filled first)
                popped_buffers.sort(key=lambda b: b["input_ids"].size(0), reverse=True)
                for buf in popped_buffers:
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(buf, {'buffer_list': buffer_list})
                    else:
                        yield self.postprocess_buffer(buf)
                    yeild_success = True
                    yield_count += 1

            while len(buffer_list) > self.max_buffer_size:
                # Evict the buffer with the most tokens (closest to full, least padding waste)
                max_tok_idx = max(range(len(buffer_list)), key=lambda i: buffer_list[i]["input_ids"].size(0))
                logger.debug(
                    f"Failed to pack data to exactly {self.num_images_expected} images, "
                    f"yield a data sample with "
                    f"{buffer_list[max_tok_idx]['input_ids'].size(0)} tokens."
                )
                if self.strict_mode:
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(max_tok_idx)), _make_custom_infos())
                    else:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(max_tok_idx)))
                else:
                    if yield_count % sync_interval == 0:
                        yield self.postprocess_buffer(buffer_list.pop(max_tok_idx), _make_custom_infos())
                    else:
                        yield self.postprocess_buffer(buffer_list.pop(max_tok_idx))
                yeild_success = True
                yield_count += 1

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1

    # NOTE:
    # @staticmethod
    # def get_cu_seqlens_and_indexes(
    #     data_index: torch.LongTensor,  # (seq_len,)
    #     input_ids: torch.LongTensor,  # (seq_len,)
    #     labels: torch.LongTensor,  # (seq_len,)
    #     image_flags: torch.LongTensor,  # (num_images,)
    #     img_start_token_id: int,
    #     img_token_id: int,  # pylint: disable=W0613
    #     img_end_token_id: int,  # pylint: disable=W0613
    #     ignored_token_ids: List[int],
    #     len2weight: callable,
    # ):
    #     indexes = []
    #     cu_seqlens = [0]
    #     loss_weight = []
    #     image_con_flags = []

    #     start = data_index.min()
    #     end = data_index.max() + 1
    #     for i in range(start, end):
    #         num_tokens = (data_index == i).sum().item()
    #         cu_seqlens.append(cu_seqlens[-1] + num_tokens)
    #         assert num_tokens > 0

    #         # NOTE:
    #         tmp_input_ids = input_ids[cu_seqlens[i]:cu_seqlens[i+1]]
    #         tmp_img_start_shift = torch.cat([torch.zeros(1, dtype=torch.long), (tmp_input_ids == img_start_token_id).long()], dim=0)[:-1]
    #         tmp_not_img_token = (tmp_input_ids != img_token_id).long()
    #         tmp_indexes = ((tmp_img_start_shift + tmp_not_img_token).cumsum(0) - 1).tolist()
    #         indexes.extend(tmp_indexes)

    #         curr_data_index = data_index[cu_seqlens[-2] : cu_seqlens[-2] + num_tokens]
    #         assert (curr_data_index == i).all(), data_index

    #         curr_labels = labels[cu_seqlens[-2] : cu_seqlens[-2] + num_tokens]
    #         num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
    #         loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)


    #     image_con_flags = None
    #     loss_weight = torch.tensor(loss_weight, dtype=torch.float32)

    #     # validate_relation = torch.stack([input_ids, torch.tensor(indexes)], dim=1)
    #     return cu_seqlens, indexes, image_con_flags, loss_weight


    @staticmethod
    def get_cu_seqlens_and_indexes(
        data_index: torch.LongTensor,      # (seq_len,)
        input_ids: torch.LongTensor,       # (seq_len,)
        labels: torch.LongTensor,          # (seq_len,)
        image_flags: torch.LongTensor,     # (num_images,)
        img_start_token_id: int,
        img_token_id: int,
        img_end_token_id: int,
        ignored_token_ids: List[int],
        len2weight: callable,
        is_image_duplicated_for_und_flags: Optional[torch.BoolTensor] = None,
    ):
        """
        - Computes cu_seqlens and per-token indexes mapping
        - If an image is duplicated for und-branch, its entire span
          copies indexes from the immediately previous image span
        - Assumes duplicated image is adjacent and same length
        """

        seq_len = input_ids.numel()
        device = input_ids.device

        indexes = torch.empty(seq_len, dtype=torch.long, device=device)
        loss_weight = torch.empty(seq_len, dtype=torch.float32, device=device)

        counts = torch.bincount(data_index)
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device),
             counts.cumsum(0)]
        ).tolist()

        do_dup_fix = (
            is_image_duplicated_for_und_flags is not None
            and bool(is_image_duplicated_for_und_flags.any())
        )

        img_ptr = 0

        for i in range(len(cu_seqlens) - 1):
            seg_l = cu_seqlens[i]
            seg_r = cu_seqlens[i + 1]

            tmp_input_ids = input_ids[seg_l:seg_r]
            tmp_labels = labels[seg_l:seg_r]

            start_shift = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    (tmp_input_ids == img_start_token_id).long(),
                ]
            )[:-1]

            not_img = (tmp_input_ids != img_token_id).long()

            tmp_indexes = (start_shift + not_img).cumsum(0) - 1

            # Fix duplicated image spans if needed
            starts = (tmp_input_ids == img_start_token_id).nonzero(as_tuple=True)[0]

            if starts.numel() > 0:
                ends = (tmp_input_ids == img_end_token_id).nonzero(as_tuple=True)[0]
                num_imgs = starts.numel()

                if do_dup_fix:
                    local_flags = is_image_duplicated_for_und_flags[
                        img_ptr : img_ptr + num_imgs
                    ]

                    if local_flags.any():
                        dup_idx = local_flags.nonzero(as_tuple=True)[0]

                        span_len = ends - starts + 1

                        # loop only over duplicated images
                        for d in dup_idx.tolist():
                            if d == 0:
                                continue

                            p = d - 1

                            assert span_len[p] == span_len[d]

                            s0, e0 = starts[p], ends[p]
                            s1, e1 = starts[d], ends[d]

                            old_end = tmp_indexes[e1].clone()
                            tmp_indexes[s1 : e1 + 1] = tmp_indexes[s0 : e0 + 1]
                            new_end = tmp_indexes[e1]
                            diff = old_end - new_end
                            if diff != 0 and e1 + 1 < tmp_indexes.numel():
                                tmp_indexes[e1 + 1:] -= diff

                img_ptr += num_imgs

            # Write into global tensor
            indexes[seg_l:seg_r] = tmp_indexes

            num_eff = (tmp_labels != IGNORE_TOKEN_ID).sum().item()
            w = float(len2weight(num_eff))

            loss_weight[seg_l:seg_r] = w

        return cu_seqlens, indexes.flatten().tolist(), None, loss_weight


WARNING_CNT = defaultdict(int)


# NOTE:
def internevo_collate_fn(
    features,
    max_item_length: int,
    img_start_token_id: int,
    img_token_id: int,
    img_end_token_id: int,
    ignored_token_ids: List[int],
    pad_id: int = 0,
    micro_num: int = 1,
    len2weight: callable = None,
    loss_reduction_all_gather: bool = False,
    patch_size: int = 16,
):
    if not isinstance(features, list):
        features = [features]

    if len(features) > micro_num:
        raise NotImplementedError(f"{len(features)=} > {micro_num=}")

    if len(features) < micro_num and WARNING_CNT["micro_num_warning"] < 5:
        logger.warning(
            f"{len(features)=} > {micro_num=}, " f"the features will be padded to satisfy micro_num requirement"
        )
        WARNING_CNT["micro_num_warning"] += 1

    # ensure that the len(features) is equal to the required micro_num
    num_features = len(features)
    while len(features) < micro_num:
        features.append(copy.deepcopy(features[0]))
        features[-1]["labels"] = torch.full_like(features[-1]["labels"], IGNORE_TOKEN_ID)

    indexes = []
    cu_seqlens = []
    cu_num_images_list = [0]

    worker_state_key_list = []
    worker_state_dict_list = []
    worker_state_custom_infos_list = []

    num_samples = 0
    num_padding_tokens = 0

    for feat_idx, feat in enumerate(features):
        already_packed = feat.pop('already_packed', False)

        if already_packed:
            curr_cu_seqlens, curr_indexes = feat.pop('cu_seqlens'), feat.pop('indexes')
            # NOTE:
            assert img_token_id not in feat['input_ids']

            for k in ["input_ids", "labels", "type_ids"]:
                feat[k] = torch.LongTensor(feat[k])
            for k in ['pixel_values', 'image_seq_lens', 'image_flags', 'image_con_flags', 'image_for_gen_flags', 'image_for_gen_loss_flags', 'is_image_duplicated_for_und_flags']:
                feat[k] = None

            # calculate loss weight
            loss_weight = []
            for start, end in zip(curr_cu_seqlens[:-1], curr_cu_seqlens[1:]):
                num_effective_tokens = (feat['labels'][start:end] != IGNORE_TOKEN_ID).sum().item()
                loss_weight.extend([len2weight(num_effective_tokens)] * (end-start))

            assert len(loss_weight) == len(feat['labels'])
            curr_loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
            
            curr_image_con_flags = None

        else:
            # NOTE:
            data_index = feat.pop('data_index')
            curr_cu_seqlens, curr_indexes, curr_image_con_flags, curr_loss_weight = PackedDataset.get_cu_seqlens_and_indexes(
                data_index=data_index,
                input_ids=feat['input_ids'],
                labels=feat['labels'],
                image_flags=feat['image_flags'],
                is_image_duplicated_for_und_flags=feat['is_image_duplicated_for_und_flags'],
                img_start_token_id=img_start_token_id,
                img_token_id=img_token_id,
                img_end_token_id=img_end_token_id,
                ignored_token_ids=ignored_token_ids,
                len2weight=len2weight,
            )
            # shift labels and loss weights
            feat['labels'] = torch.cat(
                [
                    feat['labels'][1:],
                    torch.full(size=(1,), fill_value=IGNORE_TOKEN_ID)
                ],
                dim=-1,
            )
            curr_loss_weight = torch.cat(
                [
                    curr_loss_weight[1:],
                    torch.zeros(size=(1,), dtype=curr_loss_weight.dtype)
                ],
                dim=-1,
            )
        
        feat['loss_weight'] = curr_loss_weight

        if feat_idx < num_features:
            num_samples += len(curr_cu_seqlens) - 1

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        feat["image_con_flags"] = curr_image_con_flags
        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        worker_state_key_list.append(feat.pop("worker_state_key"))
        worker_state_dict_list.append(feat.pop("worker_state_dict"))
        worker_state_custom_infos_list.append(feat.pop("custom_infos", None))

        num_padding_tokens += (max_item_length - feat['input_ids'].size(0))
        num_images = 0 if feat['pixel_values'] is None else len(feat['pixel_values'])
        cu_num_images_list.append(cu_num_images_list[-1] + num_images)

    batch, labels = concat_pad_data_collator(features=features, max_item_length=max_item_length, pad_id=pad_id)
    loss_weight = batch.pop("loss_weight")
    loss_weight = torch.where(labels == IGNORE_TOKEN_ID, 0, loss_weight).tolist()
    loss_reduction_all_gather = loss_reduction_all_gather
    input_ids = batch.pop('input_ids')
    type_ids = batch.pop('type_ids', torch.zeros_like(input_ids))

    concat_pixel_values = batch.pop("pixel_values")
    concat_image_seq_lens = batch.pop("image_seq_lens", None)
    concat_image_flags = batch.pop("image_flags")
    concat_image_for_gen_flags = batch.pop("image_for_gen_flags")
    concat_image_for_gen_loss_flags = batch.pop("image_for_gen_loss_flags")
    concat_is_image_duplicated_for_und_flags = batch.pop("is_image_duplicated_for_und_flags", None)
    concat_image_con_flags = batch.pop("image_con_flags")

    if concat_pixel_values is not None:
        assert len(concat_pixel_values) == cu_num_images_list[-1]
        assert concat_image_flags.size(0) == cu_num_images_list[-1]
        assert concat_image_for_gen_flags.size(0) == cu_num_images_list[-1]
        assert concat_image_for_gen_loss_flags.size(0) == cu_num_images_list[-1]

        images = []
        image_seq_lens = []
        image_flags = []
        image_con_flags = []
        image_for_gen_flags = []
        image_for_gen_loss_flags = []
        is_image_duplicated_for_und_flags = []
        image_grid_hw = []
            
        for i in range(len(features)):

            # NOTE: video
            # tmp_images = concat_pixel_values[cu_num_images_list[i]:cu_num_images_list[i+1]]

            

            if isinstance(concat_pixel_values, list):
                flatten_pixel_values, grid_hw = preprocess_pixel_values(concat_pixel_values[cu_num_images_list[i]:cu_num_images_list[i+1]], patch_size)
                images.append(flatten_pixel_values)
                image_grid_hw.append(grid_hw)
            else:
                images.append(concat_pixel_values[cu_num_images_list[i]:cu_num_images_list[i+1]])

            if concat_image_seq_lens is not None:
                image_seq_lens.append(concat_image_seq_lens[cu_num_images_list[i]:cu_num_images_list[i+1]])
            else:
                image_seq_lens.append(None)
            image_flags.append(concat_image_flags[cu_num_images_list[i]:cu_num_images_list[i+1]])
            image_for_gen_flags.append(concat_image_for_gen_flags[cu_num_images_list[i]:cu_num_images_list[i+1]])
            image_for_gen_loss_flags.append(concat_image_for_gen_loss_flags[cu_num_images_list[i]:cu_num_images_list[i+1]])
            if concat_is_image_duplicated_for_und_flags is not None:
                is_image_duplicated_for_und_flags.append(concat_is_image_duplicated_for_und_flags[cu_num_images_list[i]:cu_num_images_list[i+1]])
            image_con_flags.append(None) #concat_image_con_flags[cu_num_images_list[i]:cu_num_images_list[i+1]])
    else:
        images = [None  for _ in range(len(features))]
        image_seq_lens = [None for _ in range(len(features))]
        image_flags =[None  for _ in range(len(features))]
        image_for_gen_flags =[None  for _ in range(len(features))]
        image_for_gen_loss_flags = [None  for _ in range(len(features))]
        is_image_duplicated_for_und_flags = [None  for _ in range(len(features))]
        image_con_flags = [None for _ in range(len(features))]
        image_grid_hw = [None for _ in range(len(features))]

    batch.pop("attention_mask", None)
    batch.pop("position_ids", None)
    for k in batch:
        logger.warning(f"{k=} is ignored in batch")

    batch_new = {
        "input_ids": input_ids,
        "images": images,
        "image_seq_lens": image_seq_lens,
        "image_flags": image_flags,
        "image_con_flags": image_con_flags,
        "image_for_gen_flags": image_for_gen_flags,
        "image_for_gen_loss_flags": image_for_gen_loss_flags,
        "is_image_duplicated_for_und_flags": is_image_duplicated_for_und_flags,
        "cu_seqlens": cu_seqlens,
        "indexes": indexes,
        "type_ids": type_ids,
        "num_samples": num_samples,
        "num_padding_tokens": num_padding_tokens,
        "worker_state_key_list": worker_state_key_list,
        "worker_state_dict_list": worker_state_dict_list,
        "worker_state_custom_infos_list": worker_state_custom_infos_list,
        "loss_weight": loss_weight,
        "loss_reduction_all_gather": loss_reduction_all_gather,
        "image_grid_hw": image_grid_hw
    }

    # shift label
    assert labels.ndim == 2

    return batch_new, labels
