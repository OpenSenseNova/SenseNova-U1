# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Uses `LabelSmoother` ignore-index from HuggingFace transformers
# (HuggingFace Inc., Apache-2.0).
import copy
import io
import math
import os
import cv2
import random
import re
import sys
from typing import Dict, List

import imageio
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from transformers.trainer_pt_utils import LabelSmoother

from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger
from sensenovavl.data.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    SIGLIP_MEAN,
    SIGLIP_STD,
)

# NOTE: all data is assumed to live on the local filesystem.
# Object-storage (petrel / aoss / s3://) loading paths have been removed.

logger = get_logger(__name__, logging_level="info")

FIRST_ECHO = {}
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from typing import Dict, List, Union
import math
FPS_MIN = 1
FPS_MAX = 4
FRAME_FACTOR = 1
FPS = 1.0
FPS_MIN_FRAMES = 1
FPS_MAX_FRAMES = 256
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes

def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0:
            if len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(video_path, num_frames, sample="rand", fix_start=None, min_num_frames=4):

    gif = imageio.get_reader(video_path)
    vlen = len(gif)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(video_path, num_frames, sample="rand", fix_start=None, clip=None, min_num_frames=4):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_fps = np.random.uniform(FPS_MIN, min(FPS_MAX,fps))
    ele = {
            'fps': t_fps,
            'min_frames':min_num_frames,
            'max_frames':num_frames,
        }
    nframes = smart_nframes(ele, total_frames=vlen, video_fps=fps)
    idx = np.linspace(0, vlen - 1, nframes)  # 生成均匀分布的浮点数索引
    idx = np.round(idx)  # 将浮点数索引四舍五入为最近的整数
    frame_indices = idx.astype(int).tolist()


    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    timestamps = [round(frame_idx/fps,1) for frame_idx in frame_indices]

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    ## get the real fps
    real_fps = fps
    sample_fps = len(frames)/duration

    return frames, duration, real_fps, sample_fps, timestamps


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r"_(\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    clip=None,  # pylint: disable=W0613
    min_num_frames=4,
):
    def extract_number(file_name):
        match = re.search(r"(\d+)\.[a-zA-Z]+$", file_name)
        if match:
            return int(match.group(1))
        print("Invalid File name")
        return float("inf")  # if not found any number, return inf to ensure the file is at last

    image_list = sorted(list(os.listdir(video_path)))
    frames = sorted(image_list, key=extract_number)
    frames = [os.path.join(video_path, f) for f in frames]
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    # read frames
    frames = [Image.open(f).convert("RGB") for f in frames]
    return frames


class WeightedConcatDataset(ConcatDataset):
    """WeightedConcatDataset"""

    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")


class TCSLoader(object):
    """Local image / video loader.

    Historically this wrapped an object-storage client (petrel / aoss); the
    repo now assumes all data lives on the local filesystem, so this just
    reads images and decodes videos from local paths. The name and the
    ``tcs_loader=`` keyword are kept for backward compatibility.
    """

    def __init__(self):
        pass

    def __call__(self, fn, image_type="image", max_num_frames=-1, min_num_frames=4, sample="rand", clip=None):
        if image_type == "image":
            return Image.open(fn).convert("RGB")

        elif image_type == "video":
            fps = None
            duration = None
            timestamps = None
            t_fps = None
            if fn.endswith('/'):
                frames = read_frames_folder(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            sample=sample)
            elif fn.endswith('.gif'):
                frames = read_frames_gif(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                         sample=sample)
            else:
                frames, duration, fps, t_fps, timestamps = read_frames_decord(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            sample=sample, clip=clip)
            return frames, duration, fps, t_fps, timestamps


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type="imagenet", resize=True):
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        if resize:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:  # only for native resolution
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
    else:
        if pad2square is False:  # now we use this transform function by default
            if resize:
                transform = T.Compose(
                    [
                        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )
            else: # only for native resolution
                transform = T.Compose(
                    [
                        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )

        else:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )

    return transform


def preprocess_sensenovalm_v3(
    template_name,
    data_item,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: List[int],
    text_only: bool = False,
    use_packed_ds: bool = True,
    padding: bool = False,
    truncation: bool = False,
    ds_name: str = None,
    num_image: int = 1,
    cfg_drop: bool = False,
) -> Dict:
    if "conversations" in data_item:  # mm_chat type    # NOTE:
        return preprocess_sensenovalm_v3_mm_chat(
            template_name=template_name,
            conversations=data_item["conversations"],
            tokenizer=tokenizer,
            num_image_token_list=num_image_token_list,
            text_only=text_only,
            use_packed_ds=use_packed_ds,
            padding=padding,
            truncation=truncation,
            ds_name=ds_name,
            num_image=num_image,
            cfg_drop=cfg_drop,
        )
    elif "conversation" in data_item:  # st_nlp_chat type
        return preprocess_sensenovalm_v3_st_nlp_chat(
            template_name=template_name,
            conversations=data_item["conversation"],
            tokenizer=tokenizer,
            num_image_token_list=num_image_token_list,
            text_only=text_only,
            use_packed_ds=use_packed_ds,
            padding=padding,
            truncation=truncation,
            ds_name=ds_name,
            num_image=num_image,
        )

    else:
        raise NotImplementedError("Wrong Data Format")

def preprocess_sensenovalm_v3_st_nlp_chat(
    template_name,  # pylint: disable=W0613
    conversations,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: List[int],  # pylint: disable=W0613
    text_only: bool = False,  # pylint: disable=W0613
    use_packed_ds: bool = True,  # pylint: disable=W0613
    padding: bool = False,  # pylint: disable=W0613
    truncation: bool = False,
    ds_name: str = None,  # pylint: disable=W0613
    num_image: int = 1,  # pylint: disable=W0613
):
    conv = conversations[0]
    assert "input" in conv.keys() and "output" in conv.keys()
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    suffix = "<|im_end|>"
    suffix_as_eos = True
    sep = "\n"
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    input_ids, labels = [], []
    next_needs_bos_token = False

    for single_turn_conversation in conversations:
        _input = single_turn_conversation.get("input", "")
        output = single_turn_conversation["output"]

        if _input:  # Process input for sft format
            input_encode = tokenizer.encode(_input, return_attention_mask=False, add_special_tokens=False)
            if next_needs_bos_token:
                input_ids += bos_token_id
                labels += [IGNORE_TOKEN_ID] * len(bos_token_id)
            input_ids += input_encode
            labels += [IGNORE_TOKEN_ID] * len(input_encode)
        else:
            if next_needs_bos_token:
                input_ids += bos_token_id
                labels += [IGNORE_TOKEN_ID] * len(bos_token_id)
        # Process output for both sft and pretrain formats
        output += suffix if _input else ""
        output_encode = tokenizer.encode(output, return_attention_mask=False, add_special_tokens=False)
        input_ids += output_encode
        labels += copy.deepcopy(output_encode)

        # Add EOS_TOKEN (with loss)
        if not suffix_as_eos:
            next_needs_bos_token = True
            input_ids += eos_token_id
            labels += copy.deepcopy(eos_token_id)
        else:
            next_needs_bos_token = False
        # Add SEP (without loss)
        if sep != "":
            sep_encode = tokenizer.encode(sep, return_attention_mask=False, add_special_tokens=False)
            input_ids += sep_encode
            labels += [IGNORE_TOKEN_ID] * len(sep_encode)

    if truncation and len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[: tokenizer.model_max_length]
        labels = labels[: tokenizer.model_max_length]
    input_ids = torch.LongTensor([input_ids])  # [1,N] to match the before
    targets = torch.LongTensor([labels])  # [1,N] to match the before
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_sensenovalm_v3_mm_chat(     # NOTE:
    template_name,
    conversations,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: List[int],
    text_only: bool = False,
    use_packed_ds: bool = True,  # pylint: disable=W0613
    padding: bool = False,
    truncation: bool = False,
    ds_name: str = None,
    num_image: int = 1,
    cfg_drop: bool = False, # if true, we need to make sure all text except <image> is dropped including think tag
) -> Dict:
    assert not padding
    assert template_name == "sensenovalm2-chat-v3"

    roles = {
        "human": "user",
        "system": "system",
        "knowledge": "knowledge",
        "gpt": "assistant",
        "plugin": "system name=<|plugin|>",
        "plugin-return": "environment name=<|plugin|>",
        "interpreter": "system name=<|interpreter|>",
        "interpreter-return": "environment name=<|interpreter|>",
    }
    conversation_start = "<|im_start|>"
    conversation_end = "<|im_end|>\n"
    num_image = len(num_image_token_list)
    image_token_position_random = getattr(gpc.config, "image_token_position_random", "none")
    if not text_only:
        new_conversations = []
        cur_image_idx = 0
        # NOTE:
        if image_token_position_random == "end":
            for conv in conversations:
                if conv["value"].count("<image>") == 1 and conv["value"].startswith("<image>\n"):
                    if random.random() < 0.5:
                        conv["value"] = conv["value"].replace("<image>\n", "")
                        conv["value"] = conv["value"] + "\n<image>"
        elif image_token_position_random == "random":
            for conv in conversations:
                if conv["value"].count("<image>") == 1 and conv["value"].startswith("<image>\n"):
                    if random.random() < 0.5:
                        conv["value"] = conv["value"].replace("<image>\n", "")
                        text = conv["value"]
                        pos = random.randint(0, len(text))
                        conv["value"] = text[:pos] + "\n<image>\n" + text[pos:]

        for conv in conversations:
            while "<image>" in conv["value"] and cur_image_idx < num_image:
                image_tokens = (
                    f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[cur_image_idx]}{IMG_END_TOKEN}"
                )
                conv["value"] = conv["value"].replace("<image>", image_tokens, 1)
                cur_image_idx += 1
            new_conversations.append(conv)
        conversations = new_conversations

        if cur_image_idx < num_image:
            print(
                f"WARNING: image flag mismatch: cur_image_idx {cur_image_idx} "
                f"vs. num_image {num_image}. This dataset is {ds_name}."
            )
            sys.stdout.flush()

    # Tokenize conversations
    tokens = []
    labels = []

    # export config from gpc
    thinking_method = getattr(gpc.config, "thinking_method", "sys")
    st_prompt_ratio = float(getattr(gpc.config, "st_prompt_ratio", 0.0))
    rm_all_think_token = getattr(gpc.config, "rm_all_think_token", False)
    reason_prompt = 'Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.'
    st_prompt = '你是商汤科技开发的日日新融合模态大模型，中文名叫商量，英文名叫SenseNova，是一个有用无害的人工智能助手。'

    # 通过全局conversations判断thinking flag
    is_thinking_flag = False
    for conv in conversations:
        if conv["from"] == "system":
            if reason_prompt in conv["value"]:
                is_thinking_flag = True
                break
        if conv["from"] == "gpt":
            if  conv["value"].startswith("<think>"):
                is_thinking_flag = True
                break
    
    # for qwen3 thinking method, rm all reason_prompt in data
    if thinking_method == 'tag':
        if conversations[0]["from"] == "system":
            if reason_prompt in conversations[0]["value"]:
                conversations[0]["value"] = re.sub(r'(\n?)(' + re.escape(reason_prompt) + ')', '', conversations[0]["value"])
            if conversations[0]["value"] == "":
                conversations.pop(0)

    # for intermlm thinking method, add reason_prompt to system prompt
    if thinking_method == 'sys' and is_thinking_flag:
        if conversations[0]["from"] == "system":
            if reason_prompt not in conversations[0]["value"]:
                if random.random() < 0.5:
                    conversations[0]["value"] = reason_prompt + "\n" + conversations[0]["value"]
                else:
                    conversations[0]["value"] = conversations[0]["value"] + "\n" + reason_prompt
        else:
            conversations.insert(0, {"from": "system", "value": reason_prompt})

    # add default st prompt to empty system prompt
    if st_prompt_ratio > 0 and random.random() < st_prompt_ratio:
        if conversations[0]["from"] != "system":
            conversations.insert(0, {"from": "system", "value": st_prompt})
    # rm all think token
    if rm_all_think_token:  # for 冷启动，暂时删除所有think过程
        new_conversations = []
        for conv in conversations:
            if conv["from"] == "system":
                conv["value"] = re.sub(r'(\n?)(' + re.escape(reason_prompt) + ')', '', conv["value"])
            elif conv["from"] == "gpt":
                conv["value"] = conv["value"].replace("<think>\n", "").replace("\n</think>", "")
            if conv["value"] != "":
                new_conversations.append(conv)
        conversations = new_conversations

    for _, conv in enumerate(conversations):
        if conv["from"] not in roles:
            print(f"WARNING: Unknown role, skip. {conv}")
            continue
        role = roles[conv["from"]]
        content = conv["value"]
        if role != "assistant" or (role == "assistant" and conv.get("is_input", False)):
            user_info = f"{conversation_start}{role}\n{content}{conversation_end}"
            tokenized_user = tokenizer(user_info, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            tokens.extend(tokenized_user)
            labels.extend([IGNORE_TOKEN_ID] * len(tokenized_user))
        elif role == "assistant":
            if thinking_method == 'tag' or is_thinking_flag:
                if content.startswith("<think>"):
                    assis_start = f"{conversation_start}{role}\n"
                else:
                    if cfg_drop:
                        assis_start = f"{conversation_start}{role}\n"
                    else:
                        assis_start = f"{conversation_start}{role}\n<think>\n\n</think>\n\n"
            else:
                assis_start = f"{conversation_start}{role}\n"
                
            tokens_assistant_start = tokenizer(assis_start, return_attention_mask=False, add_special_tokens=False)[
                "input_ids"
            ]
            tokens.extend(tokens_assistant_start)
            labels.extend([IGNORE_TOKEN_ID] * len(tokens_assistant_start))
            assis_info = f"{content}{conversation_end}"
            tokenized_assistant = tokenizer(assis_info, return_attention_mask=False, add_special_tokens=False)[
                "input_ids"
            ]
            tokens.extend(tokenized_assistant)
            labels.extend(copy.deepcopy(tokenized_assistant))
        else:
            print(f"Not processed role, skip. {conv}")
    if truncation and len(tokens) > tokenizer.model_max_length:
        tokens = tokens[: tokenizer.model_max_length]
        labels = labels[: tokenizer.model_max_length]
    input_ids = torch.LongTensor([tokens])  # [1,N] to match the before
    targets = torch.LongTensor([labels])  # [1,N] to match the before
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def find_closest_aspect_ratio_v0(aspect_ratio, target_ratios, width, height, image_size):
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
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def preprocess_pixel_values(pixel_values, patch_size=16):
    all_flatten_pixel_values = []
    all_grid_hw = []

    for idx, px in enumerate(pixel_values):
        c, h, w = px.shape
        grid_h = h // patch_size
        grid_w = w // patch_size

        flatten_pixel_values = (
            px.view(
                c,
                grid_h,
                patch_size,
                grid_w,
                patch_size
            )
            .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
            .reshape(grid_h * grid_w, c * patch_size**2)
        )

        all_flatten_pixel_values.append(flatten_pixel_values)
        all_grid_hw.append([grid_h, grid_w])

    all_flatten_pixel_values = torch.concat(all_flatten_pixel_values, dim=0) if len(all_flatten_pixel_values) > 0 else torch.empty((0, 0))
    all_grid_hw = torch.tensor(all_grid_hw) if len(all_grid_hw) > 0 else torch.empty((0, 2))
        
    return all_flatten_pixel_values, all_grid_hw

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

# copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60
def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 256 * 32 * 32, max_pixels: int = 16384 * 32 * 32
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def dynamic_preprocess_native_resolution(image, size_factor=32, min_pixels=4 * 32 * 32, max_pixels=16384 * 32 * 32, **kwargs):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image

def _resize_and_center_crop(pil_img: Image.Image, target_w: int, target_h: int, resample=Image.BICUBIC):
    """Resize (keeping aspect) so it fully covers target, then center-crop to (target_w, target_h)."""
    w, h = pil_img.size
    if w == 0 or h == 0:
        raise ValueError(f"Invalid image size: {pil_img.size}")

    # scale to cover
    scale = max(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = pil_img.resize((new_w, new_h), resample=resample)

    # center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def resize_crop_from_candidates(image: Image.Image, candidates, **kwargs):
    if not candidates:
        raise ValueError("`candidates` is empty.")

    resample = kwargs.get("resample", Image.BICUBIC)
    tie_break = kwargs.get("tie_break", "min_upscale")
    return_chosen = kwargs.get("return_chosen", False)

    ow, oh = image.size
    log_ratio = math.log(ow / oh)

    best = None
    best_key = None

    for cand in candidates:
        tw, th = int(cand[0]), int(cand[1])
        if tw <= 0 or th <= 0:
            continue

        aspect_err = abs(log_ratio - math.log(tw / th))

        scale = max(tw / ow, th / oh)
        new_w = ow * scale
        new_h = oh * scale
        crop_waste = (new_w * new_h) - (tw * th)

        if tie_break == "min_crop":
            secondary = crop_waste
        else:
            secondary = scale

        key = (aspect_err, secondary, tw * th)
        if best_key is None or key < best_key:
            best_key = key
            best = (tw, th)

    if best is None:
        raise ValueError("No valid (W,H) found in `candidates`.")

    out = _resize_and_center_crop(image, best[0], best[1], resample=resample)
    return (out, best) if return_chosen else out
