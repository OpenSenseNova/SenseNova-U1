# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, List

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger
from sensenovavl.data.constants import (
    IGNORE_INDEX,
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
)
from sensenovavl.data.dataset import (
    WeightedConcatDataset,
    build_transform,
    preprocess_sensenovalm_v3,
    dynamic_preprocess_native_resolution,
)

from .dataset_interleaved_iterable import (
    IGNORE_TOKEN_ID,
    PackedDataset,
    concat_pad_data_collator,
    InterleavedDataset,
    ImageTextPairDataset,
    internevo_collate_fn,
)
import mmap
from io import TextIOWrapper

from .t2i_prompts import T2I_EDITING_SYSTEM_MESSAGE
from .cfg_cond_drop_utils import *

# global llm logger
logger = get_logger(__file__)


def seconds_to_minutes_secondswithdot(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{int(minutes):02d}:{secs:05.2f}"


def seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{int(minutes):02d}:{int(secs):02d}"


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        num_image_token,
        data_rank,
        data_world_size,
        distributed_mode,
        ds_name=None,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        force_shuffle=False,
        dynamic_image_size=False,
        dynamic_image_version=None,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        max_num_frame=128,
        min_num_frame=1,
        max_dynamic_images=4,
        max_multi_image_dynamic_patch=12,
        sampling_method="rand",
        repeat_time=1,
        random_seed=0,
        normalize_type="imagenet",
        type_id=None,
        type2typeid=None,
        ### Image Generation Related
        candidate_resolutions_for_gen=None,
        cfg_txt_uncond_drop_prob=0,
        cfg_img_uncond_drop_prob=0,
        cfg_txtimg_uncond_drop_prob=0,
        cfg_is_uncond_drop_independent=True,
        enabel_und_loss=False,
        language=None,
    ):
        super().__init__()
        assert not group_by_length, "Group length not support anymore"

        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        if gpc.is_rank_for_log():
            logger.info(f"[Dataset] num_image_token: {num_image_token}")
            logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
            logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
            logger.info(
                f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
            )
            logger.info("Formatting inputs...Skip in lazy mode")

        self.ds_name = ds_name
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # parameters for distributed training
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distribued = False
        self.distributed_mode = distributed_mode

        self.dataset_type = "pair"
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length

        self.root = meta["root"]
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.force_shuffle = force_shuffle
        self.dynamic_image_size = dynamic_image_size
        self.dynamic_image_version = dynamic_image_version
        if self.dynamic_image_version is None:
            self.dynamic_image_version = "native_resolution"
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.max_dynamic_images = max_dynamic_images
        self.max_multi_image_dynamic_patch = max_multi_image_dynamic_patch

        # parameters for native resolution
        self.patch_size = gpc.config.data.get("patch_size", 16)
        self.downsample_ratio = gpc.config.data.get("down_sample_ratio", 0.5)
        self.max_pixels = gpc.config.data.get("max_pixels", 4096 * 4096)
        self.min_pixels = gpc.config.data.get("min_pixels", 256 * 256)
        self.max_pixels_gen = gpc.config.data.get("max_pixels_gen", self.max_pixels)
        self.min_pixels_gen = gpc.config.data.get("min_pixels_gen", self.min_pixels)

        self.normalize_type = normalize_type

        self._state_dict = {}

        self.meta = meta

        self.annotation_file = meta["annotation"]

        self.length = math.ceil(meta["length"] * meta["repeat_time"])

        self.repeat_time = repeat_time

        self.raw_data = []

        self.type_id = type_id
        self.type2typeid = type2typeid
        self.typeid2type = {typeid: type for type, typeid in type2typeid.items()}

        self.candidate_resolutions_for_gen = candidate_resolutions_for_gen
        self.cfg_txt_uncond_drop_prob = cfg_txt_uncond_drop_prob
        self.cfg_img_uncond_drop_prob = cfg_img_uncond_drop_prob
        self.cfg_txtimg_uncond_drop_prob = cfg_txtimg_uncond_drop_prob
        self.cfg_is_uncond_drop_independent = cfg_is_uncond_drop_independent
        self.add_gen_tokens_for_und = gpc.config.get("add_gen_tokens_for_und", False)

        self.image_context_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_START_TOKEN
        )
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.enabel_und_loss = enabel_und_loss
        self.language = language

        self._state_dict = {"file_shift": 0, "bytes_offset": 0, "line_shift": 0}

        if self.group_by_length:
            raise NotImplementedError

        if self.template_name == "sensenovalm2-chat-v3":
            preprocess_function = preprocess_sensenovalm_v3
        else:
            raise ValueError(f"template_name {self.template_name}")

        self.preprocess_function = preprocess_function
        self.dataset_replacement = gpc.config.dataset_replacement

    def load_state_dict(self, state_dict):
        self._state_dict.update(state_dict)

    def _get_mmap(self, data_path):

        assert data_path.endswith(".jsonl") or data_path.endswith(".txt")
        f = open(data_path, "rb")  # pylint: disable=consider-using-with
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as e:
            print(f"Meet ValueError when loading {data_path}", flush=True)
            raise e
        self.handles = [f, mm]
        return self.handles[-1]

    def __len__(self):
        return self.length

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def rand_offset(self, margin):
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    def _fixed_scale_size(
        self,
        size,
        scale,
    ):
        """Rescale a size by a ratio.

        Args:
            size (tuple[int]): (w, h).
            scale (float | tuple(float)): Scaling factor.

        Returns:
            tuple[int]: scaled size.
        """
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        w, h = size
        # don't need o.5 offset
        return int(w * float(scale[0])), int(h * float(scale[1]))

    def check_t2i_data(self, data_item):
        if len(data_item["conversations"]) != 2:
            return False
        if (
            not data_item["conversations"][1]["value"].endswith("<image>")
            or data_item["conversations"][1]["value"].count(("<image>")) != 1
        ):
            return False
        if any(
            keyword in data_item["conversations"][0]["value"]
            for keyword in [
                IMG_CONTEXT_TOKEN,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                "<image>",
            ]
        ):
            return False
        if any(
            keyword in data_item["conversations"][1]["value"]
            for keyword in [IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]
        ):
            return False
        return True

    def check_it2i_data(self, data_item):
        if len(data_item["conversations"]) != 2:
            return False
        if (
            not data_item["conversations"][1]["value"].endswith("<image>")
            or data_item["conversations"][1]["value"].count(("<image>")) != 1
        ):
            return False
        if data_item["conversations"][0]["value"].count("<image>") != (
            len(data_item["image"]) - 1
        ):
            return False
        if any(
            keyword in data_item["conversations"][0]["value"]
            for keyword in [IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]
        ):
            return False
        if any(
            keyword in data_item["conversations"][1]["value"]
            for keyword in [IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]
        ):
            return False
        return True

    def check_i2t_data(self, data_item, image_path_list):
        image_count = 0
        for conv in data_item["conversations"]:
            if conv["from"] == "human":
                if any(
                    keyword in conv["value"]
                    for keyword in [IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]
                ):
                    return False, image_count
                image_count += conv["value"].count("<image>")
            # model round
            elif conv["from"] == "gpt":
                if any(
                    keyword in conv["value"]
                    for keyword in [
                        IMG_CONTEXT_TOKEN,
                        IMG_START_TOKEN,
                        IMG_END_TOKEN,
                        "<image>",
                    ]
                ):
                    return False, image_count
        return True, image_count

    def check_pure_text_data(self, data_item):
        for conv in data_item["conversations"]:
            if any(
                keyword in conv["value"]
                for keyword in [
                    IMG_CONTEXT_TOKEN,
                    IMG_START_TOKEN,
                    IMG_END_TOKEN,
                    "<image>",
                ]
            ):
                return False
        return True

    def image_gen_prepare_conv(self, data_item, task_type, image_path_list):
        image_for_gen_flags = []
        image_for_gen_loss_flags = []
        is_image_duplicated_for_und_flags = []
        interleave_not_repeat_last_flag = False
        is_cfg_drop_txt = False

        human_round_i = 0
        for i in range(len(data_item["conversations"])):
            if data_item["conversations"][i]["from"] == "human":
                human_round_i = i
                break

        if task_type == "mm_t2i":
            valid_data = self.check_t2i_data(data_item)
            if not valid_data:
                assert False, f"invalid data, {data_item}"

            drop_prob = float(self.cfg_txt_uncond_drop_prob)
            if random.random() < drop_prob:
                data_item["conversations"][human_round_i]["value"] = ""
                # remove possible thinking process prior to the generated image
                data_item["conversations"][-1]["value"] = "<image>"
                is_cfg_drop_txt = True
            else:
                data_item["conversations"].insert(
                    0, {"from": "system", "value": T2I_EDITING_SYSTEM_MESSAGE}
                )

            image_for_gen_flags = [1]
            image_for_gen_loss_flags = [1]
            is_image_duplicated_for_und_flags = [0]

            # disable this
            # if self.enabel_und_loss:
            #     # duplicate the gen image for the und branch with small probability so the model learns to predict <|im_end|>
            #     if random.random() < 0.1:
            #         image_for_gen_flags.append(0)
            #         image_for_gen_loss_flags.append(0)
            #         is_image_duplicated_for_und_flags.append(1)
            #         image_path_list.append(image_path_list[-1])
            #         data_item['conversations'][-1]['value'] = data_item['conversations'][-1]['value'].replace('<image>', '<image><image>')

        elif task_type == "mm_it2i":
            valid_data = self.check_it2i_data(data_item)
            if not valid_data:
                assert False, f"invalid data, {data_item}"

            if self.cfg_is_uncond_drop_independent:
                text_drop_prob = float(self.cfg_txt_uncond_drop_prob)
                if random.random() < text_drop_prob:
                    data_item["conversations"][human_round_i]["value"] = "<image>" * (
                        len(image_path_list) - 1
                    )
                    # remove possible thinking process prior to the generated image
                    data_item["conversations"][-1]["value"] = "<image>"
                    is_cfg_drop_txt = True

                img_drop_prob = float(self.cfg_img_uncond_drop_prob)
                if random.random() < img_drop_prob:
                    data_item["conversations"][human_round_i]["value"] = (
                        data_item["conversations"][human_round_i]["value"]
                        .replace("<image>\n", "")
                        .replace("<image>", "")
                    )
                    image_path_list = image_path_list[-1:]
                elif not is_cfg_drop_txt:
                    data_item["conversations"].insert(
                        0, {"from": "system", "value": T2I_EDITING_SYSTEM_MESSAGE}
                    )
            else:
                drop_prob = sum(
                    [
                        self.cfg_txt_uncond_drop_prob,
                        self.cfg_img_uncond_drop_prob,
                        self.cfg_txtimg_uncond_drop_prob,
                    ]
                )
                if random.random() < drop_prob:
                    drop_case = random.choices(
                        ["txt", "img", "txtimg"],
                        weights=[
                            self.cfg_txt_uncond_drop_prob,
                            self.cfg_img_uncond_drop_prob,
                            self.cfg_txtimg_uncond_drop_prob,
                        ],
                        k=1,
                    )[0]

                    if drop_case == "txt":
                        data_item["conversations"][human_round_i]["value"] = (
                            "<image>" * (len(image_path_list) - 1)
                        )
                        # remove possible thinking process prior to the generated image
                        data_item["conversations"][-1]["value"] = "<image>"
                        is_cfg_drop_txt = True
                    elif drop_case == "img":
                        data_item["conversations"][human_round_i]["value"] = (
                            data_item["conversations"][human_round_i]["value"]
                            .replace("<image>\n", "")
                            .replace("<image>", "")
                        )
                        image_path_list = image_path_list[-1:]
                    else:
                        data_item["conversations"][human_round_i]["value"] = ""
                        image_path_list = image_path_list[-1:]
                        # remove possible thinking process prior to the generated image
                        data_item["conversations"][-1]["value"] = "<image>"
                        is_cfg_drop_txt = True
                else:
                    data_item["conversations"].insert(
                        0, {"from": "system", "value": T2I_EDITING_SYSTEM_MESSAGE}
                    )

            image_for_gen_flags = [0] * (len(image_path_list) - 1) + [1]
            image_for_gen_loss_flags = [0] * (len(image_path_list) - 1) + [1]
            is_image_duplicated_for_und_flags.extend([0] * len(image_path_list))

            # disable this
            # if self.enabel_und_loss:
            #     # duplicate the gen image for the und branch with small probability so the model learns to predict <|im_end|>
            #     if random.random() < 0.1:
            #         image_for_gen_flags.append(0)
            #         image_for_gen_loss_flags.append(0)
            #         is_image_duplicated_for_und_flags.append(1)
            #         image_path_list.append(image_path_list[-1])
            #         data_item['conversations'][-1]['value'] = data_item['conversations'][-1]['value'].replace('<image>', '<image><image>')

        elif task_type == "mm_interleave_gen":
            valid_data = self.check_interleave_data(data_item, image_path_list)
            if not valid_data:
                assert False, f"invalid data, {data_item}"

            drop_prob = sum(
                [self.cfg_txt_uncond_drop_prob, self.cfg_txtimg_uncond_drop_prob]
            )
            if random.random() < drop_prob:
                drop_case = random.choices(
                    ["txt", "txtimg"],
                    weights=[
                        self.cfg_txt_uncond_drop_prob,
                        self.cfg_txtimg_uncond_drop_prob,
                    ],
                    k=1,
                )[0]
                if drop_case == "txt":
                    for turn_i, conv in enumerate(data_item["conversations"]):
                        image_count = conv["value"].count("<image>")
                        conv["value"] = "<image>" * image_count
                    if data_item["conversations"][0]["from"] == "system":
                        data_item["conversations"].pop(0)
                    is_cfg_drop_txt = True
                else:
                    # randomly choose a gen image and only modeling it in this sample
                    gen_image_index_list = []
                    image_index_accum = 0
                    for turn_i, conv in enumerate(data_item["conversations"]):
                        if conv["from"] != "gpt":
                            image_count = conv["value"].count("<image>")
                            image_index_accum += image_count
                        else:
                            image_count = conv["value"].count("<image>")
                            for _ in range(image_count):
                                gen_image_index_list.append(image_index_accum)
                                image_index_accum += 1

                    random_gen_image_index = random.choice(gen_image_index_list)
                    data_item["conversations"] = data_item["conversations"][:2]
                    data_item["conversations"][0]["value"] = ""
                    data_item["conversations"][0]["from"] = "human"
                    data_item["conversations"][1]["value"] = "<image>"
                    data_item["conversations"][1]["from"] = "gpt"
                    image_path_list = [image_path_list[random_gen_image_index]]
                    is_cfg_drop_txt = True

            image_path_list_new = []
            image_i = 0
            for turn_i, conv in enumerate(data_item["conversations"]):
                # non gpt round
                if conv["from"] != "gpt":
                    image_count = conv["value"].count("<image>")
                    if image_count > 0:
                        image_path_list_new.extend(
                            image_path_list[image_i : image_i + image_count]
                        )
                        image_i += image_count
                        image_for_gen_flags.extend([0] * image_count)
                        image_for_gen_loss_flags.extend([0] * image_count)
                        is_image_duplicated_for_und_flags.extend([0] * image_count)
                # gpt turn, we duplicate the generated image for the understanding branch
                else:
                    image_count = conv["value"].count("<image>")
                    if image_count > 0:
                        if turn_i != len(data_item["conversations"]) - 1:
                            conv["value"] = conv["value"].replace(
                                "<image>", "<image><image>"
                            )
                            image_for_gen_flags.extend([1, 0] * image_count)
                            image_for_gen_loss_flags.extend([1, 0] * image_count)
                            is_image_duplicated_for_und_flags.extend(
                                [0, 1] * image_count
                            )
                            for cur_image_i in range(image_count):
                                image_path_list_new.extend(
                                    [
                                        image_path_list[image_i + cur_image_i],
                                        image_path_list[image_i + cur_image_i],
                                    ]
                                )
                            image_i += image_count
                        else:
                            # last turn
                            image_for_gen_flags.extend([1, 0] * image_count)
                            image_for_gen_loss_flags.extend([1, 0] * image_count)
                            is_image_duplicated_for_und_flags.extend(
                                [0, 1] * image_count
                            )
                            for cur_image_i in range(image_count):
                                image_path_list_new.extend(
                                    [
                                        image_path_list[image_i + cur_image_i],
                                        image_path_list[image_i + cur_image_i],
                                    ]
                                )
                            if (
                                not self.enabel_und_loss
                                or (
                                    conv["value"].endswith("<image>")
                                    and random.random() > 0.2
                                )
                                or is_cfg_drop_txt
                            ):
                                # do not repeat the last gen image
                                interleave_not_repeat_last_flag = True
                                conv["value"] = repeat_all_but_last(
                                    conv["value"], "<image>"
                                )
                                image_for_gen_flags = image_for_gen_flags[:-1]
                                image_for_gen_loss_flags = image_for_gen_loss_flags[:-1]
                                is_image_duplicated_for_und_flags = (
                                    is_image_duplicated_for_und_flags[:-1]
                                )
                                image_path_list_new = image_path_list_new[:-1]
                            else:
                                conv["value"] = conv["value"].replace(
                                    "<image>", "<image><image>"
                                )
                            image_i += image_count
            image_path_list = image_path_list_new
        else:
            is_image_duplicated_for_und_flags = [0] * len(image_path_list)
            for conv in data_item["conversations"]:
                if conv["from"] == "gpt" and "<image>" in conv["value"]:
                    assert (
                        False
                    ), f"invalid data with <image> in model turn, {data_item}"

        return (
            data_item,
            image_path_list,
            image_for_gen_flags,
            image_for_gen_loss_flags,
            is_image_duplicated_for_und_flags,
            interleave_not_repeat_last_flag,
            is_cfg_drop_txt,
        )

    def image_gen_append_image_size_info(
        self, data_item, task_type, image_for_gen_flags, image_for_gen_resolutions
    ):
        if task_type == "mm_t2i":
            data_item["conversations"][0][
                "value"
            ] += f"\nThe resolution of the image should be {image_for_gen_resolutions[0]}"
        else:
            raise NotImplementedError("Only support t2i task")

        return data_item

    def multi_modal_get_item(self, data_item):  # NOTE:
        # for debug
        # data_item = {"image": ["image1.png"]*8, "conversations": [{"from": "human", "value": "A<image>A<image>A"}, {"from": "gpt", "value": "B<image>B<image>"}, {"from": "human", "value": "A<image>"}, {"from": "gpt", "value": "B<image>"}, {"from": "human", "value": "A"}, {"from": "gpt", "value": "<image><image>"}]}
        # self.type_id = 5

        if (
            self.dynamic_image_size
            and self.dynamic_image_version == "native_resolution"
        ):
            assert (
                not self.pad2square
            ), "pad2square is not supported for native resolution"
            transform = build_transform(
                is_train=self.is_train,
                input_size=self.image_size,
                pad2square=self.pad2square,
                resize=False,
            )
        else:
            transform = build_transform(
                is_train=self.is_train,
                input_size=self.image_size,
                pad2square=self.pad2square,
            )

        task_type = self.typeid2type[self.type_id]
        is_image_gen_task = task_type in ["mm_t2i", "mm_it2i", "mm_interleave_gen"]

        if "image" in data_item:
            if isinstance(data_item["image"], list):
                image_path_list = data_item["image"]
            else:
                image_path_list = [data_item["image"]]
        else:
            image_path_list = data_item["images"]

        original_image_path_length = len(image_path_list)

        # we process the prompts and optinally do drop conditions for image generation task
        if is_image_gen_task:
            (
                data_item,
                image_path_list,
                image_for_gen_flags,
                image_for_gen_loss_flags,
                is_image_duplicated_for_und_flags,
                interleave_not_repeat_last_flag,
                is_cfg_drop_txt,
            ) = self.image_gen_prepare_conv(
                deepcopy(data_item), task_type, deepcopy(image_path_list)
            )
        else:
            is_cfg_drop_txt = False
            image_for_gen_flags = [0] * len(image_path_list)
            image_for_gen_loss_flags = [0] * len(image_path_list)
            is_image_duplicated_for_und_flags = [0] * len(image_path_list)

        # we only append <image> for non image gen data
        if len(image_path_list) == 1 and not is_image_gen_task:
            for conv in data_item["conversations"]:
                if conv["from"] == "human":
                    # the first round for human should have an image
                    if "<image>" not in conv["value"]:
                        conv["value"] = "<image>\n" + conv["value"]
                    break

        if not is_image_gen_task:
            valid_data, img_count = self.check_i2t_data(data_item, image_path_list)
            if not valid_data:
                assert False, f"invalid data, {data_item}"
            if img_count == 0:
                return self.pure_text_get_item(data_item)
            if img_count != len(image_path_list):
                assert False, f"invalid data, {data_item}"
        assert len(image_for_gen_flags) == len(
            image_path_list
        ), f"Mismatch between number of image files ({len(image_path_list)}) and <image> token ({len(image_for_gen_flags)})"
        if is_image_gen_task:
            assert sum(
                image_for_gen_flags
            ), "image generation task but no <image> in model's turn"
        max_pixels_gen = self.max_pixels_gen
        min_pixels_gen = self.min_pixels_gen
        images, num_tiles = [], []
        image_for_gen_resolutions = []
        num_image = len(image_path_list)
        for image_i, (image_path, is_image_for_gen) in enumerate(
            zip(image_path_list, image_for_gen_flags)
        ):
            if (
                is_image_gen_task
                and image_i > 0
                and is_image_duplicated_for_und_flags[image_i]
            ):
                prev_patch = images[-1]
                images.append(prev_patch)
                image_for_gen_resolutions.append(image_for_gen_resolutions[-1])
                num_tiles.append(num_tiles[-1])
                continue

            try:
                if self.root is not None:
                    image_path = os.path.join(self.root, image_path)

                if self.tcs_loader is not None:
                    image = self.tcs_loader(image_path)
                else:
                    image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(
                    f"Fail to Load Image, {self.ds_name}: {self.root} {image_path} for exception: {e}"
                )

            ## for some editing sample, we have to ensure the input and output image has same size
            if (
                image_i == 0
                and original_image_path_length == 2
                and task_type == "mm_it2i"
                and str(data_item.get("must_same_size", "false")).lower() == "true"
            ):
                image = image.resize((data_item["width"][-1], data_item["height"][-1]))

            if self.dynamic_image_size:
                if self.dynamic_image_version == "native_resolution":
                    if is_image_gen_task:
                        assert (
                            self.candidate_resolutions_for_gen is not None
                        ), "candidate_resolutions_for_gen is None"
                        patch = dynamic_preprocess_native_resolution(
                            image,
                            min_pixels=min_pixels_gen,
                            max_pixels=(
                                max_pixels_gen
                                if num_image == 1
                                else max(
                                    min(
                                        max_pixels_gen,
                                        (self.max_tokens - 3072) * 32 * 32 // num_image,
                                    ),
                                    min_pixels_gen,
                                )
                            ),
                            size_factor=int(self.patch_size / self.downsample_ratio),
                        )
                    else:
                        patch = dynamic_preprocess_native_resolution(
                            image,
                            min_pixels=self.min_pixels,
                            max_pixels=(
                                self.max_pixels
                                if num_image == 1
                                else max(
                                    min(
                                        self.max_pixels,
                                        (self.max_tokens - 3072) * 32 * 32 // num_image,
                                    ),
                                    self.min_pixels,
                                )
                            ),
                            size_factor=int(self.patch_size / self.downsample_ratio),
                        )
                    images.append(patch)
                    w, h = patch.size
                    image_for_gen_resolutions.append((w, h))
                    num_tiles.append(
                        int(w * h // self.patch_size**2 * self.downsample_ratio**2)
                    )
                else:
                    raise NotImplementedError(
                        f"dynamic_image_version must be 'native_resolution', got {self.dynamic_image_version!r}"
                    )
            else:
                images.append(image)
                num_tiles.append(1)

        if (
            self.dynamic_image_size
            and self.dynamic_image_version == "native_resolution"
        ):
            pixel_values = [transform(image) for image in images]
        else:
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            assert sum(num_tiles) == len(
                pixel_values
            ), f"{sum(num_tiles)=}, {len(pixel_values)=}"
        num_patches = len(pixel_values)

        if not self.dynamic_image_size:
            assert all(
                num_patches == 1 for num_patches in num_tiles
            ), f"The number of patches should be 1, but got {num_tiles}."

        if (
            self.dynamic_image_size
            and self.dynamic_image_version == "native_resolution"
        ):
            num_image_tokens = [num_tile for num_tile in num_tiles]
        else:
            num_image_tokens = [
                self.num_image_token * num_tile for num_tile in num_tiles
            ]
        if self.template_name in ["sensenovalm2-chat", "sensenovalm2-chat-v3"]:
            ret = self.preprocess_function(
                self.template_name,
                deepcopy(data_item),
                self.tokenizer,
                num_image_tokens,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=num_image,
                cfg_drop=is_cfg_drop_txt,
            )
        else:
            ret = self.preprocess_function(
                self.template_name,
                [deepcopy(data_item["conversations"])],
                self.tokenizer,
                num_image_tokens,
                group_by_length=self.group_by_length,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=num_image,
            )

        assert (ret["input_ids"][0] == self.image_context_token_id).sum() == sum(
            num_image_tokens
        ), f"image tokens are truncated, this dataset is {self.ds_name} {(ret['input_ids'][0] ==self.image_context_token_id).sum()} != {sum(num_image_tokens)}"

        if self.dynamic_image_version == "native_resolution":
            assert isinstance(
                pixel_values, list
            ), f"pixel_values should be a list, but got {type(pixel_values)}"

        if task_type in ["mm_t2i", "mm_it2i"]:
            image_start_token_positions = (
                (ret["input_ids"][0] == self.image_start_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
            image_end_token_positions = (
                (ret["input_ids"][0] == self.image_end_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )

            if not self.enabel_und_loss:
                ret["labels"][0][:] = IGNORE_INDEX
            else:
                assert not is_image_duplicated_for_und_flags[
                    -1
                ], "never predict <img_end> in t2i and it2i for now"
                # no need to predict img_start for now
                ret["labels"][0][image_start_token_positions[-1] :] = IGNORE_INDEX

        elif task_type == "mm_interleave_gen":
            image_start_token_positions = (
                (ret["input_ids"][0] == self.image_start_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
            image_end_token_positions = (
                (ret["input_ids"][0] == self.image_end_token_id)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )

            if not self.enabel_und_loss or is_cfg_drop_txt:
                ret["labels"][0][:] = IGNORE_INDEX
            else:
                for image_i in range(len(image_for_gen_flags)):
                    cur_image_for_gen_flag = image_for_gen_flags[image_i]
                    cur_is_image_duplicated_for_und_flag = (
                        is_image_duplicated_for_und_flags[image_i]
                    )
                    cur_image_start_idx = image_start_token_positions[image_i]
                    cur_image_end_idx = image_end_token_positions[image_i]

                    if cur_image_for_gen_flag:
                        ret["labels"][0][
                            cur_image_start_idx + 1 : cur_image_end_idx + 1
                        ] = IGNORE_INDEX
                    if cur_is_image_duplicated_for_und_flag:
                        ret["labels"][0][
                            cur_image_start_idx : cur_image_end_idx + 1
                        ] = IGNORE_INDEX

                if interleave_not_repeat_last_flag:
                    ret["labels"][0][image_end_token_positions[-1] :] = IGNORE_INDEX

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            image_for_gen_flags=torch.tensor(image_for_gen_flags, dtype=torch.bool),
            image_for_gen_loss_flags=torch.tensor(
                image_for_gen_loss_flags, dtype=torch.bool
            ),
            is_image_duplicated_for_und_flags=torch.tensor(
                is_image_duplicated_for_und_flags, dtype=torch.bool
            ),
        )

        return ret

    def video_get_item(self, data_item):
        if self.dynamic_image_version == "native_resolution":
            assert (
                not self.pad2square
            ), "pad2square is not supported for native resolution"
            transform = build_transform(
                is_train=self.is_train,
                input_size=self.image_size,
                pad2square=self.pad2square,
                resize=False,
            )
        else:
            transform = build_transform(
                is_train=self.is_train,
                input_size=self.image_size,
                pad2square=self.pad2square,
            )

        for i, conv in enumerate(data_item["conversations"]):
            if conv["from"] == "human":
                if "<video>" not in conv["value"]:
                    data_item["conversations"][i]["value"] = (
                        "<video>\n" + data_item["conversations"][i]["value"]
                    )
                break

        video_file = data_item["video"]
        video_path = (
            os.path.join(self.root, video_file) if self.root is not None else video_file
        )

        clip = data_item.get("clip", None)
        image_list, video_duration, video_fps, sample_fps, sample_timestamp = (
            self.tcs_loader(
                video_path,
                image_type="video",
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
                clip=clip,
            )
        )
        if video_duration:
            temporal_instruction = f"The time range of this video is [00:00-{seconds_to_minutes_seconds(video_duration)}], and the following is a series of {len(image_list)} frames sampled at {round(sample_fps,1)} FPS.\n"
            assert len(image_list) == len(sample_timestamp)
            special_tokens = "\n".join(
                [
                    f"{seconds_to_minutes_secondswithdot(frame_time)}]:<image>"
                    for i, frame_time in enumerate(sample_timestamp)
                ]
            )
            special_tokens = temporal_instruction + special_tokens
        else:
            special_tokens = "\n".join(
                ["Frame{}:<image>".format(i + 1) for i in range(len(image_list))]
            )

        if data_item["conversations"][0]["from"] == "human":
            data_item["conversations"][0]["value"] = data_item["conversations"][0][
                "value"
            ].replace("<video>\n", special_tokens)
        else:
            data_item["conversations"][1]["value"] = data_item["conversations"][1][
                "value"
            ].replace("<video>\n", special_tokens)

        raw_images = []
        num_patches_per_image = []
        pixel_values = []
        num_image_tokens = []

        for image in image_list:
            raw_images.append(image)
            if self.dynamic_image_size and len(image_list) <= self.max_dynamic_images:
                if self.dynamic_image_version == "native_resolution":
                    patch = dynamic_preprocess_native_resolution(
                        image,
                        max_pixels=(
                            self.max_pixels
                            if len(image_list) == 1
                            else self.max_pixels * 2 // len(image_list)
                        ),
                        size_factor=int(self.patch_size / self.downsample_ratio),
                    )
                    patches = [patch]
                    w, h = patch.size
                    num_image_tokens.append(
                        int(w * h // self.patch_size**2 * self.downsample_ratio**2)
                    )
                else:
                    raise NotImplementedError
            else:
                if (
                    self.dynamic_image_size
                    and self.dynamic_image_version == "native_resolution"
                ):
                    patch = dynamic_preprocess_native_resolution(
                        image,
                        max_pixels=512**2,
                        size_factor=int(self.patch_size / self.downsample_ratio),
                    )
                    patches = [patch]
                    w, h = patch.size
                    num_image_tokens.append(
                        int(w * h // self.patch_size**2 * self.downsample_ratio**2)
                    )
                else:
                    patches = [image]
            num_patches_per_image.append(len(patches))
            pixel_values.extend([transform(patch) for patch in patches])
        if self.dynamic_image_version != "native_resolution":
            pixel_values = torch.stack(pixel_values)
        num_patches = len(pixel_values)

        if self.dynamic_image_version != "native_resolution":
            num_image_tokens = [
                self.num_image_token * num_patches
                for num_patches in num_patches_per_image
            ]
        if self.template_name in ["sensenovalm2-chat", "sensenovalm2-chat-v3"]:
            ret = self.preprocess_function(
                self.template_name,
                deepcopy(data_item),
                self.tokenizer,
                num_image_tokens,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=num_patches,
            )
        else:
            ret = self.preprocess_function(
                self.template_name,
                [deepcopy(data_item["conversations"])],
                self.tokenizer,
                num_image_tokens,
                group_by_length=self.group_by_length,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=num_patches,
            )

        assert (ret["input_ids"][0] == self.image_context_token_id).sum() == sum(
            num_image_tokens
        ), (
            f"video tokens are truncated, this dataset is {self.ds_name} "
            + "{(ret['input_ids'][0] ==self.image_context_token_id).sum()} != {sum(num_image_tokens)}"
        )

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            image_for_gen_flags=torch.tensor([0] * num_patches, dtype=torch.bool),
            image_for_gen_loss_flags=torch.tensor([0] * num_patches, dtype=torch.bool),
            is_image_duplicated_for_und_flags=torch.tensor(
                [0] * num_patches, dtype=torch.bool
            ),
        )
        return ret

    def check_interleave_data(self, data_item, image_path_list):
        if len(image_path_list) > 40:
            return False
        image_count = 0
        image_count_for_gen = 0
        for conv in data_item["conversations"]:
            if any(
                keyword in conv["value"]
                for keyword in [IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN]
            ):
                return False
            image_count += conv["value"].count("<image>")
            if conv["from"] == "gpt":
                image_count_for_gen += image_count
        if image_count != len(image_path_list):
            return False
        if image_count_for_gen == 0:
            return False
        return True

    def pure_text_get_item(self, data_item):  # NOTE:
        valid_data = self.check_pure_text_data(data_item)
        if not valid_data:
            assert False, f"invalid data, {data_item}"

        if self.dynamic_image_version == "native_resolution":
            pixel_values = []
            num_patches = 0
        else:
            images = [
                Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255)),
            ]
            transform = build_transform(
                is_train=self.is_train,
                input_size=self.image_size,
                pad2square=self.pad2square,
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            num_patches = len(pixel_values)
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        if self.template_name in ["sensenovalm2-chat", "sensenovalm2-chat-v3"]:
            ret = self.preprocess_function(
                self.template_name,
                deepcopy(data_item),
                self.tokenizer,
                [None],
                text_only=True,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=0,
            )
        else:
            ret = self.preprocess_function(
                self.template_name,
                [deepcopy(data_item["conversations"])],
                self.tokenizer,
                [None],
                text_only=True,
                group_by_length=self.group_by_length,
                use_packed_ds=True,
                ds_name=self.ds_name,
                num_image=0,
            )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            image_for_gen_flags=torch.tensor([0] * num_patches, dtype=torch.bool),
            image_for_gen_loss_flags=torch.tensor([0] * num_patches, dtype=torch.bool),
            is_image_duplicated_for_und_flags=torch.tensor(
                [0] * num_patches, dtype=torch.bool
            ),
            source=ret.get("source", ""),
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distribued
            and self.worker_id is not None
        ):
            self.worker_distribued = True

            if self.data_rank == 0:
                logger.info(
                    f"worker_distributed is enabled, {self.num_workers=}, dataset length {self.length}"
                )

    def get_sample(self, line) -> Dict[str, torch.Tensor]:

        try:
            line = line.decode("utf-8")
            data_item = json.loads(line)
        except Exception as e:
            logger.info(
                f"[{self.ds_name}] [Worker id {self.worker_id}] Encounting Error with {type(e)} \n Error message: {e}"
            )
            return None
        try:
            if "image" in data_item or "images" in data_item:
                ret = self.multi_modal_get_item(data_item)
            elif (
                "video" in data_item
                and data_item["video"] is not None
                and data_item["video"] != ""
            ):
                ret = self.video_get_item(data_item)
            else:
                ret = self.pure_text_get_item(data_item)
            # skip all turncated data
            ### TODO comment to ignore skip, cut in split buffer
            if len(ret["input_ids"]) > self.max_tokens:
                raise ValueError(
                    f"Too Long Sample, Skip: {self.ds_name}  [Worker id {self.worker_id}] "
                    f"[Length {len(ret['input_ids'])}]"
                )
            if (ret["labels"] == IGNORE_TOKEN_ID).all() and self.typeid2type[
                self.type_id
            ] not in ["mm_t2i", "mm_it2i", "mm_interleave_gen"]:
                # do not have
                sample_conv = "|-|".join(
                    [c["value"] for c in data_item["conversations"]]
                )
                raise ValueError(
                    f"Do not have valid value in sample target in dataset {self.ds_name} {sample_conv}"
                )

            if (
                len(ret["input_ids"]) == 0
                or len(ret["labels"]) == 0
                or len(ret["input_ids"]) != len(ret["labels"])
            ):
                # do not have
                # sample_conv = '|-|'.join([c['value'] for c in data_item['conversations']])
                raise ValueError(
                    f"Empty data {self.ds_name}  [Worker id {self.worker_id}] {data_item}"
                )

            # add type ids

            ret["type_ids"] = torch.zeros_like(ret["input_ids"]) + self.type_id

        except Exception as e:
            if isinstance(e, IndexError):
                logger.exception(e)
            # import traceback; traceback.print_exc()
            # if gpc.is_rank_for_log():
            if self.ds_name not in [
                "coyo1_caption",
                "coyo2_caption",
                "laion400m1_caption",
                "laion400m2_caption",
                "laioncoco_caption",
                "WikiWeb2M-train-single-image",
                "WikiWeb2M-train-multi-image",
                "wukong_ocr_zh_bak",
            ]:
                data_path = "null"
                tmp_root = "" if self.root is None else self.root
                if "image" in data_item:
                    if isinstance(data_item["image"], list):
                        data_path = [
                            os.path.join(tmp_root, item) for item in data_item["image"]
                        ]
                    else:
                        data_path = os.path.join(tmp_root, data_item["image"])

                logger.info(
                    f"[{self.ds_name}] [Worker id {self.worker_id}] skip data with image {data_path} for exception(first 200 char): {str(e)[:200]}"
                )

            return None

        return ret

    def reset(
        self,
    ):
        # reset
        self._state_dict = {
            "file_shift": 0,
            "bytes_offset": 0,
            "line_shift": 0,  # 用于整体的文件行数计数
        }

    def __iter__(self):  # NOTE:
        self._enable_worker_distributed()

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        self.worker_id = worker_id = num_workers * self.data_rank + worker_id
        self.num_workers = num_workers = num_workers * self.data_world_size

        self.worker_state_key = f"work_state_{self.worker_id}"

        repeat_time = math.ceil(self.repeat_time)  # 文件需要打开的次数

        file_shift = self._state_dict["file_shift"]

        if file_shift >= repeat_time:
            # the dataset should have been used up
            self.reset()
            raise StopIteration

        for file_idx in range(file_shift, repeat_time):
            file: TextIOWrapper = self._get_mmap(self.annotation_file)
            file.seek(0, 0)
            relative_offset = self._state_dict["bytes_offset"] - file.tell()
            assert (
                relative_offset >= 0
            ), f"Invalid offset {relative_offset} in file {self.annotation_file}"
            file.seek(relative_offset, 1)

            line_offset = -1  # 用于文件内部计数
            for line_offset, line in enumerate(iter(file.readline, b"")):
                old_bytes_offset = self._state_dict["bytes_offset"]
                old_line_shift = self._state_dict["line_shift"]

                self._state_dict["bytes_offset"] = file.tell()
                self._state_dict["line_shift"] += 1
                file.madvise(mmap.MADV_DONTNEED, 0, file.tell())

                if old_line_shift % num_workers == worker_id:
                    # next_sample = self.get_sample(line)
                    # if next_sample is not None:
                    #     next_sample['meta_info'] = deepcopy(self._state_dict)
                    #     yield next_sample
                    # yield string instead of sample to save memory usage
                    if self.get_sample(line):
                        yield line

                if (
                    self._state_dict["line_shift"] >= self.length
                    and not self.dataset_replacement
                ):
                    break

            # if gpc.is_rank_for_log():
            #     logger.info(f"[{worker_id}] dataset {self.ds_name} datafile {self.annotation_file} has been used up for {file_idx} time & the end line is {self._state_dict['line_shift']}")

            self._state_dict["file_shift"] += 1
            self._state_dict["bytes_offset"] = 0

            if (
                self._state_dict["line_shift"] >= self.length
                and not self.dataset_replacement
            ):
                break

        # if gpc.is_rank_for_log():
        logger.info(f"[{worker_id}] dataset {self.ds_name} has been ran out of!!!")

        self.reset()
        raise StopIteration


def get_dataset_type_ids_map(ds_collections, type_id_offset):
    ds_name_list = list(ds_collections.keys())
    mm_type_id_version = gpc.config.data.get("mm_type_id_version", "v0")
    ds2typeid = {}
    type2typeid = {}
    if mm_type_id_version == "v0":
        type2typeid = {
            "multimodal": type_id_offset,
            "nlp_sft": type_id_offset + 1,
            "mm_interleaved": type_id_offset + 2,
            "mm_t2i": type_id_offset + 3,
            "mm_it2i": type_id_offset + 4,
            "mm_interleave_gen": type_id_offset + 5,
        }
        for ds_name in ds_name_list:
            if ds_collections[ds_name].get("dataset_type", None) == "interleaved":
                ds2typeid[ds_name] = type2typeid["mm_interleaved"]

            elif ds_collections[ds_name].get("task", None) is not None:
                task_name = f"mm_{ds_collections[ds_name]['task']}"
                if task_name not in type2typeid:
                    new_values = max(list(type2typeid.values())) + 1
                    type2typeid.update({task_name: new_values})
                ds2typeid[ds_name] = type2typeid[task_name]

            elif (
                ds_collections[ds_name]["root"] or ds_collections[ds_name]["root"] == ""
            ):
                ds2typeid[ds_name] = type2typeid["multimodal"]
            else:
                assert (
                    ds_collections[ds_name]["root"] is None
                ), "should be None for nlp sft data"
                ds2typeid[ds_name] = type2typeid["nlp_sft"]
    else:
        raise NotImplementedError

    return ds2typeid, type2typeid


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    num_image_token,
    data_rank,
    data_world_size,
    distributed_mode,
    group_by_length=False,
    force_shuffle=False,
    dynamic_image_version=None,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    min_num_frame=8,
    max_num_frame=24,
    data_augment=True,
    type_id_offset=0,
):
    datasets = []
    lengths = []

    if gpc.is_rank_for_log():
        print(
            f"Please note...  max_num_frame: {max_num_frame}, min_num_frame: {min_num_frame}"
        )

    with open(data_args.meta_path) as file:
        ds_collections = json.load(file)

    ds2typeid, type2typeid = get_dataset_type_ids_map(
        ds_collections, type_id_offset=type_id_offset
    )

    paired_data = []
    paired_data_length = []
    cc_data = []
    cc_data_length = []
    plain_pair_data = []
    plain_pair_data_length = []

    if group_by_length:
        group_by_length = False
        if gpc.is_rank_for_log():
            logger.info(f"set group_by_length to False")

    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]["repeat_time"]

        if gpc.is_rank_for_log():
            logger.info(f"Add dataset:{ds_name}")

        if ds_collections[ds_name].get("dataset_type", None) == "interleaved":
            dataset_cls = InterleavedDataset
            # override some parameters
            dataset = dataset_cls(
                template_name="sensenovalm2-plain",  # use plain style for cc data
                meta=ds_collections[ds_name],
                tokenizer=tokenizer,
                tcs_loader=tcs_loader,
                ds_name=ds_name,
                data_rank=data_rank,
                data_world_size=data_world_size,
                num_image_token=num_image_token,
                image_size=data_args.force_image_size,
                is_train=False,
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,  # for simplicity, we disable dynamic image size for interleaved dataset
                dynamic_image_version=dynamic_image_version,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=(
                    max_dynamic_patch
                    if "max_dynamic_patch" not in ds_collections[ds_name]
                    else ds_collections[ds_name]["max_dynamic_patch"]
                ),
                max_num_images=data_args.num_images_expected,
                image_switch_prob=ds_collections[ds_name].get("image_switch_prob", 0),
                type_id=ds2typeid[ds_name],
                repeat_time=ds_collections[ds_name]["repeat_time"],  # 新增
            )
            cc_data.append(dataset)
            cc_data_length.append(dataset.dataset_weight)

        elif ds_collections[ds_name].get("dataset_type", None) == "plain_pair":
            dataset_cls = ImageTextPairDataset
            dataset = dataset_cls(
                template_name="sensenovalm2-plain",  # use plain style for cc data
                meta=ds_collections[ds_name],
                tokenizer=tokenizer,
                tcs_loader=tcs_loader,
                ds_name=ds_name,
                data_rank=data_rank,
                data_world_size=data_world_size,
                num_image_token=num_image_token,
                image_size=data_args.force_image_size,
                is_train=False,
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=False,  # for simplicity, we disable dynamic image size for laion dataset, since usually it's now the first stage
                use_thumbnail=False,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                max_num_images=data_args.num_images_expected,
                image_switch_prob=ds_collections[ds_name].get("image_switch_prob", 0),
                type_id=ds2typeid[ds_name],
            )
            plain_pair_data.append(dataset)
            plain_pair_data_length.append(dataset.dataset_weight)

        else:
            # NOTE:
            dataset_cls = LazySupervisedDataset
            dataset = dataset_cls(
                template_name=data_args.conv_style,
                meta=ds_collections[ds_name],
                tokenizer=tokenizer,
                tcs_loader=tcs_loader,
                data_rank=data_rank,
                data_world_size=data_world_size,
                distributed_mode=distributed_mode,
                ds_name=ds_name,
                num_image_token=num_image_token,
                image_size=data_args.force_image_size,
                is_train=data_augment,
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                force_shuffle=force_shuffle,
                dynamic_image_size=dynamic_image_size,
                dynamic_image_version=dynamic_image_version,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=(
                    max_dynamic_patch
                    if "max_dynamic_patch" not in ds_collections[ds_name]
                    else ds_collections[ds_name]["max_dynamic_patch"]
                ),
                min_num_frame=min_num_frame,
                max_num_frame=max_num_frame,
                repeat_time=ds_collections[ds_name]["repeat_time"],
                random_seed=ds_idx,
                type_id=ds2typeid[ds_name],
                type2typeid=type2typeid,
                candidate_resolutions_for_gen=getattr(
                    data_args, "candidate_resolutions_for_gen", None
                ),
                cfg_txt_uncond_drop_prob=getattr(
                    data_args, "cfg_txt_uncond_drop_prob", 0
                ),
                cfg_img_uncond_drop_prob=getattr(
                    data_args, "cfg_img_uncond_drop_prob", 0
                ),
                cfg_txtimg_uncond_drop_prob=getattr(
                    data_args, "cfg_txtimg_uncond_drop_prob", 0
                ),
                enabel_und_loss=getattr(data_args, "enabel_und_loss", False),
                language=ds_collections[ds_name].get("language", None),
            )
            paired_data.append(dataset)
            paired_data_length.append(len(dataset))

        if gpc.is_rank_for_log():
            if hasattr(dataset, "__len__"):
                logger.info(
                    f"build dataset finished:{ds_name} with length: {len(dataset)}"
                )
            else:
                logger.info(
                    f"build dataset finished:{ds_name} with sampling weight: {dataset.dataset_weight}"
                )

    return (
        paired_data,
        cc_data,
        plain_pair_data,
        paired_data_length,
        cc_data_length,
        plain_pair_data_length,
        list(type2typeid.keys()),
    )


def image_pair_collator(
    features,
    max_item_length: int,
    img_start_token_id: int,
    img_token_id: int,
    img_end_token_id: int,
    ignored_token_ids: List[int],
    pad_id=0,
):
    if not isinstance(features, list):
        features = [features]

    # TODO: input_ids will be pad when len(features) > 1
    assert len(features) == 1

    indexes = []
    cu_seqlens = []

    num_padding_tokens = 0
    for feat in features:
        data_index = torch.ones_like(feat["input_ids"])
        curr_cu_seqlens, curr_indexes, curr_image_con_flags, _ = (
            PackedDataset.get_cu_seqlens_and_indexes(
                data_index=data_index,
                input_ids=feat["input_ids"],
                labels=None,  # FIXME: fix it.
                image_flags=feat["image_flags"],
                img_start_token_id=img_start_token_id,
                img_token_id=img_token_id,
                img_end_token_id=img_end_token_id,
                ignored_token_ids=ignored_token_ids,
                len2weight=None,  # FIXME: fix it.
            )
        )

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        feat["image_con_flags"] = curr_image_con_flags
        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        num_padding_tokens += max_item_length - feat["input_ids"].size(0)

    batch, labels = concat_pad_data_collator(
        features=features, max_item_length=max_item_length, pad_id=pad_id
    )
    input_ids = batch.pop("input_ids")
    images = [batch.pop("pixel_values")]
    image_flags = [batch.pop("image_flags")]
    image_con_flags = [batch.pop("image_con_flags")]

    batch.pop("attention_mask", None)
    batch.pop("position_ids", None)
    for k in batch:
        logger.warning(f"{k=} is ignored in batch")

    batch_new = {
        "input_ids": input_ids,
        "images": images,
        "image_flags": image_flags,
        "image_con_flags": image_con_flags,
        "cu_seqlens": cu_seqlens,
        "indexes": indexes,
        "type_ids": torch.zeros_like(input_ids),
        "num_padding_tokens": num_padding_tokens,
        # 'worker_state_key_list': worker_state_key_list,
        # 'worker_state_dict_list': worker_state_dict_list,
    }

    assert labels.ndim == 2
    labels = torch.cat(
        [
            labels[:, 1:],
            torch.full(size=(labels.size(0), 1), fill_value=IGNORE_TOKEN_ID),
        ],
        dim=1,
    )
    return batch_new, labels
