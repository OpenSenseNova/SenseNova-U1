# --------------------------------------------------------
# SenseNovaVL — derived from InternVL (OpenGVLab, MIT).
# Copyright (c) 2023 OpenGVLab. Licensed under MIT.
# Copyright (c) SenseNovaLM contributors. Modifications licensed under Apache-2.0.
# --------------------------------------------------------
import warnings
from typing import List, Optional

import torch.utils.checkpoint
import torch.distributed as dist
import math
from torch import nn
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.model.builder import create_model
from sensenovalm.model.moe.utils import SenseNovaVLMoEOutput
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.parallel import is_using_isp
from sensenovavl.utils.utils import build_abs_positions_from_grid_hw
from sensenovavl.model.modules.fm_modules import TimestepEmbedder

from .configuration_sensenovavl_chat import SenseNovaVLChatConfig
from .modeling_neo_vit import NEOVisionModel
from .utils import (
    gather_forward_split_backward,
    get_split_size,
    split_forward_gather_backward,
)

logger = get_logger(__name__)


# NOTE:
import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import and_masks, create_block_mask, or_masks
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")


def global_all_reduce_loss(local_sum, local_cnt) -> torch.Tensor:
    dist.all_reduce(local_cnt, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
    return local_sum / local_cnt.clamp_min(1)


def calculate_pad_length(seqlen, div_num):
    """
    calculate the min padlen,  make (seqlen + padlen) can be divisible by div_num

    :param seqlen: int, 序列长度
    :param div_num: int, 整数
    :return: int, 最小填充长度
    """
    if seqlen % div_num == 0:
        return 0
    else:
        padlen = div_num - (seqlen % div_num)
        return padlen

def _offsets_to_doc_ids_tensor(offsets, has_pad, split_size=1024):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]

    if has_pad:
        tmp_counts = counts[:-1]
        last_counts = counts[-1]

        num_split = last_counts // split_size 
        remainder = last_counts % split_size

        split_counts = [split_size] * num_split
        if remainder > 0:
            split_counts.append(remainder)

        counts = torch.cat([tmp_counts, torch.LongTensor(split_counts).to(dtype=tmp_counts.dtype, device=tmp_counts.device)], dim=-1)

    return torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), counts)

def create_flex_mask_padding(document_ids, modality_indicators, div_num, compile_mask):
    '''
    Current version:
    1. document attention
    2. within each document, causal attention. Within a same image, full attention
    seqlen padded to divisable by some number
    '''
    slen = document_ids.size(-1)
    padlen = calculate_pad_length(seqlen=slen, div_num=div_num)
    if padlen > 0:
        pad_doc_id = document_ids.max() + 1
        document_ids = F.pad(document_ids, (0, padlen), value=pad_doc_id)
        modality_indicators = F.pad(modality_indicators, (0, padlen), value=-1)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    def samedoc_mask(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] > 0
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        return is_image & (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=slen + padlen,
        KV_LEN=slen + padlen,
        BLOCK_SIZE=128,
        _compile=compile_mask,
    )

    return block_mask, padlen


@torch.no_grad()
def dense_from_flex_mask(mask_fn, slen, device="cuda"):
    q = torch.arange(slen, device=device)
    k = torch.arange(slen, device=device)

    Q, K = torch.meshgrid(q, k, indexing="ij")

    dummy_b = torch.zeros(1, device=device)  # important
    dummy_h = torch.zeros(1, device=device)  # not used but required

    M = mask_fn(dummy_b, dummy_h, Q, K)

    return M.bool()

def create_flex_mask_padding_image_gen(document_ids, modality_indicators, image_gen_indicators, token_pos, dup_boundary, div_num, compile_mask, mask_image_gen_tokens=False):
    slen = document_ids.size(-1)
    padlen = calculate_pad_length(seqlen=slen, div_num=div_num)
    if padlen > 0:
        pad_doc_id = document_ids.max() + 1
        document_ids = F.pad(document_ids, (0, padlen), value=pad_doc_id)
        modality_indicators = F.pad(modality_indicators, (0, padlen), value=-1)
        image_gen_indicators = F.pad(image_gen_indicators, (0, padlen), value=False)
        dup_boundary = F.pad(dup_boundary, (0, padlen), value=False)

        last = token_pos.max()
        token_pos = F.pad(token_pos, (0, padlen), value=last + 1)

    padded_len = slen + padlen
    assert document_ids.numel() == padded_len, (document_ids.numel(), padded_len)
    assert modality_indicators.numel() == padded_len, (modality_indicators.numel(), padded_len)
    assert image_gen_indicators.numel() == padded_len, (image_gen_indicators.numel(), padded_len)
    assert token_pos.numel() == padded_len, (token_pos.numel(), padded_len)
    assert dup_boundary.numel() == padded_len, (dup_boundary.numel(), padded_len)

    def causal_mask(b, h, q_idx, kv_idx):
        # causal by original position within a doc
        return token_pos[q_idx] >= token_pos[kv_idx]

    def samedoc_mask(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] > 0
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        return is_image & (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)

    # forbid any token attending to image_gen tokens except same-image image_gen tokens or formids image_gen tokens attending to corresponding duplicated image tokens ---
    def gen_kv_gate_mask(b, h, q_idx, kv_idx):
        kv_is_gen = image_gen_indicators[kv_idx]          # bool tensor
        q_is_gen  = image_gen_indicators[q_idx]           # bool tensor

        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        same_img = (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc

        gate1 = (~kv_is_gen) | (q_is_gen & same_img)

        gate2 = (~q_is_gen) | kv_is_gen | (token_pos[kv_idx] != token_pos[q_idx])

        return gate1 & gate2

    mask_mod = and_masks(mask_mod, gen_kv_gate_mask)

    def dup_boundary_gate_mask(b, h, q_idx, kv_idx):
        kv_is_dup_boundary = dup_boundary[kv_idx]
        return ~kv_is_dup_boundary
    
    mask_mod = and_masks(mask_mod, dup_boundary_gate_mask)

    if mask_image_gen_tokens:
        def gen_isolation_mask(b, h, q_idx, kv_idx):
            kv_is_gen = image_gen_indicators[kv_idx]
            return ~kv_is_gen
        mask_mod = and_masks(mask_mod, gen_isolation_mask)
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=slen + padlen,
        KV_LEN=slen + padlen,
        BLOCK_SIZE=128,
        _compile=compile_mask,
    )

    return block_mask, padlen


def build_dup_boundary(modality_indicators, gen_tok, dup_tok):
    dup_boundary = torch.zeros_like(gen_tok)

    boundary = (modality_indicators == -1)

    # dup <img_start>: boundary[i] and dup image ctx at i+1
    dup_boundary[:-1] |= boundary[:-1] & dup_tok[1:]

    # gen <img_end> right before dup <img_start>:
    # boundary[i] & gen_tok[i-1] & boundary[i+1] & dup_tok[i+2]
    dup_boundary[1:-2] |= boundary[1:-2] & gen_tok[:-3] & boundary[2:-1] & dup_tok[3:]

    return dup_boundary


def get_image_seq_lens(grid_hw: torch.Tensor, merge_size: int):
    image_seq_lens = (grid_hw[:, 0] * grid_hw[:, 1] // (merge_size**2)).tolist()
    return [int(seq_len) for seq_len in image_seq_lens]


def slice_tensor_by_image_lens(tensor: Optional[torch.Tensor], full_image_seq_lens, kept_image_seq_lens, dim: int = 0):
    if tensor is None:
        return None
    if list(full_image_seq_lens) == list(kept_image_seq_lens):
        return tensor

    chunks = []
    offset = 0
    for full_len, keep_len in zip(full_image_seq_lens, kept_image_seq_lens):
        if keep_len < 0 or keep_len > full_len:
            raise ValueError(f"Invalid keep_len={keep_len} for full_len={full_len}")
        if keep_len > 0:
            chunks.append(tensor.narrow(dim, offset, keep_len))
        offset += full_len

    if chunks:
        return torch.cat(chunks, dim=dim)
    return tensor.narrow(dim, 0, 0)


def align_selected_to_image_seq_lens(selected_mask: torch.Tensor, modality_indicators: torch.Tensor, image_seq_lens):
    flat_selected = selected_mask.reshape(-1).bool()
    modality_indicators = modality_indicators.reshape(-1)
    aligned_selected = torch.zeros_like(flat_selected)
    prompt_image_seq_lens = []
    kept_image_seq_lens = []

    for image_idx, image_seq_len in enumerate(image_seq_lens, start=1):
        image_positions = torch.nonzero((modality_indicators == image_idx) & flat_selected, as_tuple=False).flatten()
        prompt_image_seq_len = int(image_positions.numel())
        keep_len = min(prompt_image_seq_len, int(image_seq_len))
        prompt_image_seq_lens.append(prompt_image_seq_len)
        kept_image_seq_lens.append(keep_len)
        if keep_len > 0:
            aligned_selected[image_positions[:keep_len]] = True

    return aligned_selected.view_as(selected_mask), prompt_image_seq_lens, kept_image_seq_lens


def build_modality_indicators_from_context_runs(image_context_mask: torch.Tensor):
    flat_context = image_context_mask.reshape(-1).bool()
    modality_indicators = torch.full(
        (flat_context.shape[0],),
        -1,
        dtype=torch.long,
        device=flat_context.device,
    )
    if not flat_context.any():
        return modality_indicators, []

    prev_context = torch.cat(
        [torch.zeros(1, dtype=torch.bool, device=flat_context.device), flat_context[:-1]],
        dim=0,
    )
    run_start_flags = flat_context & (~prev_context)
    modality_indicators = run_start_flags.long().cumsum(0)
    modality_indicators[~flat_context] = -1

    num_runs = int(modality_indicators.max().item())
    prompt_image_seq_lens = torch.bincount(
        modality_indicators[flat_context],
        minlength=num_runs + 1,
    )[1:].tolist()
    prompt_image_seq_lens = [int(x) for x in prompt_image_seq_lens]
    return modality_indicators, prompt_image_seq_lens


def summarize_image_seq_mismatch(prompt_image_seq_lens, expected_image_seq_lens):
    prompt_image_seq_lens = [int(x) for x in prompt_image_seq_lens]
    expected_image_seq_lens = [int(x) for x in expected_image_seq_lens]
    max_len = max(len(prompt_image_seq_lens), len(expected_image_seq_lens))
    first_mismatch_idx = -1
    prompt_len = -1
    expected_len = -1
    mismatched_images = 0

    for idx in range(max_len):
        prompt_val = prompt_image_seq_lens[idx] if idx < len(prompt_image_seq_lens) else -1
        expected_val = expected_image_seq_lens[idx] if idx < len(expected_image_seq_lens) else -1
        if prompt_val != expected_val:
            mismatched_images += 1
            if first_mismatch_idx == -1:
                first_mismatch_idx = idx
                prompt_len = prompt_val
                expected_len = expected_val

    return mismatched_images, first_mismatch_idx, prompt_len, expected_len


def pack_two_branch_sequence(
    hidden_states: torch.Tensor,          # [1, L, C]
    indexes: torch.Tensor,                # [L, 3]
    document_ids: torch.Tensor,           # [L] int32
    modality_indicators: torch.Tensor,    # [L] long; -1 for non-img else image_id
    image_gen_indicators: torch.Tensor,   # [1, L] bool or [L] bool
    dup_boundary: torch.Tensor,   # [1, L] bool or [L] bool
):
    assert hidden_states.ndim == 3 and hidden_states.shape[0] == 1
    L = hidden_states.shape[1]
    assert indexes.shape[0] == L
    assert document_ids.shape[0] == L
    assert modality_indicators.shape[0] == L
    assert dup_boundary.shape[0] == L

    gen = image_gen_indicators
    if gen.ndim == 2:
        assert gen.shape[0] == 1 and gen.shape[1] == L
        gen = gen.view(-1)
    else:
        assert gen.shape[0] == L
    gen = gen.to(dtype=torch.bool)

    # new order = [gen tokens..., other tokens...]
    gen_idx = torch.nonzero(gen, as_tuple=True)[0]
    oth_idx = torch.nonzero(~gen, as_tuple=True)[0]
    perm = torch.cat([gen_idx, oth_idx], dim=0)  # (new_pos -> old_pos), shape [L]

    # inverse perm for scatter-back
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(L, device=perm.device, dtype=perm.dtype)

    packed_hidden_states = hidden_states[:, perm, :]
    packed_indexes = indexes[perm]
    packed_document_ids = document_ids[perm]
    packed_modality_indicators = modality_indicators[perm]
    packed_image_gen_indicators = gen[perm]
    packed_dup_boundary = dup_boundary[perm]

    token_pos = packed_indexes[:, 0].clone()

    return (
        packed_hidden_states,
        packed_indexes,
        packed_document_ids,
        packed_modality_indicators,
        packed_image_gen_indicators,
        perm,
        inv_perm,
        token_pos,
        packed_dup_boundary
    )


def unpack_two_branch_sequence(
    packed_output: torch.Tensor,  # [1, L, C]
    inv_perm: torch.Tensor,       # [L] (old_pos -> new_pos)
):
    """
    Scatter packed_output back to original token order.
    """
    assert packed_output.ndim == 3 and packed_output.shape[0] == 1
    L = packed_output.shape[1]
    assert inv_perm.shape[0] == L

    output = packed_output[:, inv_perm, :]
    return output

# NOTE:


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size, assuming square window

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H * W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


class SenseNovaVLChatMoTModel(PreTrainedModel):
    """SenseNovaVL Chat MoT Model."""

    config_class = SenseNovaVLChatConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: SenseNovaVLChatConfig,
        vision_model=None,
        language_model=None,
        first: bool = False,
        last: bool = False,
    ):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = config.llm_config.vocab_size
        self.img_context_token_id = config.img_context_token_id
        self.img_start_token_id = config.img_start_token_id
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.image_fold = config.image_fold
        self.ps_version = config.ps_version
        self.tp_mode = gpc.config.parallel.tensor.mode
        self.tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.image_gen_loss_weight = config.image_gen_loss_weight
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.enable_vit_sp = gpc.config.parallel.tensor.enable_vit
        self.pure_llm = gpc.config.model.pure_llm
        self.dynamic_image_version = gpc.config.data.get('dynamic_image_version', 'v3')

        self.timestep_shift = config.timestep_shift
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        self.noise_scale_max_value = config.noise_scale_max_value

        self.noise_scale = config.noise_scale
        self.P_mean = config.P_mean
        self.P_std = config.P_std
        self.t_eps = config.t_eps

        if first is True and self.pure_llm is False:
            if gpc.is_rank_for_log():
                logger.info(f"num_image_token: {self.num_image_token}")
                logger.info(f"ps_version: {self.ps_version}")
            if vision_model is not None:
                self.vision_model = vision_model
            else:
                self.vision_model = NEOVisionModel(config.vision_config)

            vision_model_mot_gen = NEOVisionModel(config.vision_config)
            # vision_model_mot_gen.embeddings.add_pos_embedding = False
            # image geneneration related modules
            patch_size = self.config.vision_config.patch_size
            merge_size = int(1 / self.downsample_ratio)
            output_dim = 3*(patch_size*merge_size)**2
            
            timestep_embedder = TimestepEmbedder(llm_hidden_size)
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, output_dim, bias=True),
            )

            self.fm_modules = nn.ModuleDict(
                {
                    "timestep_embedder": timestep_embedder,
                    "fm_head": fm_head,
                    "vision_model_mot_gen": vision_model_mot_gen
                }
            )

            if self.add_noise_scale_embedding:
                noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
                self.fm_modules['noise_scale_embedder'] = noise_scale_embedder



        if first is False or (not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)):
            pipeline_kwargs = config.llm_config.copy()
            pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
            pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
            pipeline_kwargs["pipeline_size"] = max(1, pipeline_size - 1)
            pipeline_kwargs["pipeline_rank"] = max(0, pipeline_rank - 1)
            if pipeline_kwargs["moe_location"] != "llm":
                pipeline_kwargs["moe_kwargs"] = None

            if language_model is not None:
                self.language_model = language_model
            else:
                self.language_model = create_model(gpc.config.model_type, pipeline_kwargs)

    def get_per_image_pos_ids(self, grid_hw):
        device = grid_hw.device
        max_num_patch_per_side = 32
        merge_size = int(1 / self.downsample_ratio)

        h, w = grid_hw.tolist()

        h_tok = h // merge_size
        w_tok = w // merge_size

        # spatial index k in [0, h_tok*w_tok):
        # row = k // w_tok, col = k % w_tok, pos_id = row*pos_table_side + col
        k = torch.arange(h_tok * w_tok, device=device, dtype=torch.long)
        row = k // w_tok
        col = k - row * w_tok
        spatial_ids = row * max_num_patch_per_side + col  # [h_tok*w_tok]

        return spatial_ids

    def _calculate_dynamic_mu(self, image_seq_len: int) -> float:
        denom = self.max_image_seq_len - self.base_image_seq_len
        if denom == 0:
            return float(self.base_shift)
        m = (self.max_shift - self.base_shift) / denom
        b = self.base_shift - m * self.base_image_seq_len
        return float(image_seq_len) * m + b

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int) -> torch.Tensor:
        # Apply shift on noise weight (sigma) to align with Flux behavior.
        sigma = 1 - t
        if self.time_schedule == "standard":
            shift = self.timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif self.time_schedule == "dynamic":
            mu = self._calculate_dynamic_mu(image_seq_len)
            mu_t = t.new_tensor(mu)
            if self.time_shift_type == "exponential":
                shift = torch.exp(mu_t)
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            elif self.time_shift_type == "linear":
                sigma = mu_t / (mu_t + (1 / sigma - 1))
            else:
                raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {self.time_schedule}")
        return 1 - sigma

    def prepare_image_gen_targets(self, pixel_values, image_for_gen_flags, grid_hw):
        if sum(image_for_gen_flags) == 0:
            return pixel_values, None, None, None, None, None

        patch_size = self.config.vision_config.patch_size
        merge_size = round(1 / self.downsample_ratio) 
        patch_size_after_downsample = patch_size * merge_size
        pixel_values = pixel_values.view(-1, 3*self.patch_size*self.patch_size)

        assert len(pixel_values) == (grid_hw[:, 0] * grid_hw[:, 1]).sum()
        assert len(image_for_gen_flags) == len(grid_hw)

        # only include images for generation
        image_gen_x = []
        image_gen_z = []
        image_gen_t = []
        image_gen_pos_ids = []
        image_gen_noise_scale = []

        # for all images
        pixel_values_updated = []

        
        und_mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        und_std  = torch.tensor([0.229, 0.224, 0.225],  device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)

        image_token_accum = 0
        for image_i in range(len(image_for_gen_flags)):
            cur_image_h = grid_hw[image_i, 0]
            cur_image_w = grid_hw[image_i, 1]
            cur_image_token_num = cur_image_h * cur_image_w
            if image_for_gen_flags[image_i]:
                cur_pixel_values = pixel_values[image_token_accum:image_token_accum+cur_image_token_num].clone()
                cur_pixel_values = cur_pixel_values.view(-1, 3, self.patch_size, self.patch_size)
                cur_pixel_values = (cur_pixel_values * und_std + und_mean).clamp(0, 1).view(-1, 3*self.patch_size*self.patch_size)
                cur_pixel_values = (cur_pixel_values - 0.5) * 2
                noise_scale = self.noise_scale
                image_seq_len = cur_image_token_num // (merge_size**2)
                if self.noise_scale_mode in ("resolution", "dynamic"):
                    base = float(self.noise_scale_base_image_seq_len)
                    scale = math.sqrt(image_seq_len/base)
                    noise_scale = scale * float(self.noise_scale)
                cur_noise = torch.randn_like(cur_pixel_values) * noise_scale

                u = torch.normal(mean=0.0, std=1.0, size=(1,), device=pixel_values.device) * self.P_std + self.P_mean
                t = (1 / (1 + torch.exp(-u))).to(dtype=pixel_values.dtype, device=pixel_values.device)

                # Synchronize random noise and timestep across TP ranks to ensure
                # identical flow-matching targets for the same image on all ranks.
                if self.tp_size > 1:
                    tp_group = gpc.get_group(ParallelMode.TENSOR)
                    tp_ranks = gpc.get_ranks_in_group(ParallelMode.TENSOR)
                    dist.broadcast(cur_noise, src=tp_ranks[0], group=tp_group)
                    dist.broadcast(t, src=tp_ranks[0], group=tp_group)

                t = self._apply_time_schedule(t, image_seq_len)

                t_expanded = t.expand(cur_image_token_num)
                t_expanded_merged = t.expand(cur_image_token_num // merge_size**2)

                cur_image_gen_z = t_expanded.view(-1, 1) * cur_pixel_values + (1 - t_expanded.view(-1, 1)) * cur_noise
                pixel_values_updated.append(cur_image_gen_z)

                image_gen_t.append(t_expanded_merged)
                image_gen_noise_scale.append(torch.full_like(t_expanded_merged, noise_scale/self.noise_scale_max_value))

                # for prediction after patch merged
                cur_pixel_values_reshape = cur_pixel_values.view(cur_image_h//merge_size, merge_size, cur_image_w//merge_size, merge_size, 3, patch_size, patch_size)
                cur_pixel_values_reshape = torch.einsum("h a w b c i j -> h w a i b j c", cur_pixel_values_reshape).contiguous().view(-1, patch_size_after_downsample**2*3)

                cur_image_gen_z_reshape = cur_image_gen_z.view(cur_image_h//merge_size, merge_size, cur_image_w//merge_size, merge_size, 3, patch_size, patch_size)
                cur_image_gen_z_reshape = torch.einsum("h a w b c i j -> h w a i b j c", cur_image_gen_z_reshape).contiguous().view(-1, patch_size_after_downsample**2*3)

                image_gen_x.append(cur_pixel_values_reshape)
                image_gen_z.append(cur_image_gen_z_reshape)

                per_image_pos = self.get_per_image_pos_ids(grid_hw[image_i])
                image_gen_pos_ids.append(per_image_pos)
            else:
                cur_pixel_values = pixel_values[image_token_accum:image_token_accum+cur_image_token_num]
                pixel_values_updated.append(cur_pixel_values)

            image_token_accum += cur_image_token_num

        pixel_values_updated = torch.cat(pixel_values_updated, 0)
        image_gen_x = torch.cat(image_gen_x, 0)
        image_gen_z = torch.cat(image_gen_z, 0)
    
        image_gen_t = torch.cat(image_gen_t, 0).view(-1,)
        image_gen_noise_scale = torch.cat(image_gen_noise_scale, 0).view(-1,)
        image_gen_pos_ids = torch.cat(image_gen_pos_ids, 0).view(-1,)

        image_gen_v = (image_gen_x - image_gen_z) / (1 - image_gen_t.view(-1, 1)).clamp_min(self.t_eps)
        
        return pixel_values_updated, image_gen_z, image_gen_v, image_gen_t, image_gen_pos_ids, image_gen_noise_scale

    def build_image_gen_indicators(
        self,
        modality_indicators: torch.Tensor,   # [L], long; -1 for non-img, else image id
        image_for_gen_flags: torch.Tensor,   # [N_img], bool; per-image
        image_id_base: int = 1,              # 1 if modality uses {1..N}, 0 if {0..N-1}
    ):

        assert image_for_gen_flags.dtype == torch.bool

        out = torch.zeros_like(modality_indicators, dtype=torch.bool)

        img_mask = modality_indicators >= 0
        if not img_mask.any():
            return out

        img_ids = modality_indicators[img_mask] - image_id_base

        valid = (img_ids >= 0) & (img_ids < image_for_gen_flags.numel())
        if valid.any():
            out_idx = img_mask.nonzero(as_tuple=True)[0][valid]
            out[out_idx] = image_for_gen_flags[img_ids[valid]]

        return out

    def forward(  # pylint: disable=W0102
        self,
        hidden_states=None,
        cls_embeds=None,
        image_flags: Optional[torch.LongTensor] = None,
        image_con_flags: Optional[torch.LongTensor] = None,
        image_for_gen_flags: Optional[torch.BoolTensor] = None,
        is_image_duplicated_for_und_flags: Optional[torch.BoolTensor] = None,
        images=[],  # 28072, 768
        input_ids=None,  # 1, 8192  (26)   image_token_id 151669
        cu_seqlens=None,  # 28 (-1, -1 pad)
        indexes=None,   # 8192
        inference_params=None,
        type_ids=None,
        **kwargs,
    ):
        pad_dummy_image_gen = gpc.config.get("pad_dummy_image_gen", False)
        pad_dummy_image_num = 128

        grid_hw = kwargs.get("image_grid_hw", [])   # 26
        image_seq_lens = kwargs.get("image_seq_lens", None)
        moe_losses, moe_z_losses, moe_coef_losses = [], [], []
        moe_outputs: List[SenseNovaVLMoEOutput] = []
        mtp_outputs = None
        pure_text = False
        if image_flags is None or (len(image_flags) == 1 and image_flags[0] is None) or image_flags[0].sum() == 0:
            pure_text = True
        if image_flags is not None and self.pure_llm is True:
            assert (len(image_flags) == 1 and image_flags[0] is None) or image_flags[
                0
            ].sum() == 0, "when only train llm, the data should be pure text."

        if hasattr(self, "vision_model"):
            if pure_text:
                if self.dynamic_image_version == 'native_resolution':
                    segments = int(1 / self.downsample_ratio)
                    images = [
                        torch.rand(
                            segments**2,
                            3 * self.patch_size * self.patch_size,
                            device=get_current_device(),
                            dtype=gpc.config.model.dtype,
                        )
                    ]
                    grid_hw = [torch.tensor([[segments, segments]], device=get_current_device())]
                else:
                    images = [
                        torch.rand(
                            1,
                            3,
                            self.image_size,
                            self.image_size,
                            device=get_current_device(),
                            dtype=gpc.config.model.dtype,
                        )
                    ]
                image_flags = [torch.tensor([0], dtype=torch.long, device=get_current_device())]
                image_for_gen_flags = [torch.tensor([0], dtype=torch.long, device=get_current_device())]
                is_image_duplicated_for_und_flags = [torch.tensor([0], dtype=torch.long, device=get_current_device())]

            assert isinstance(images, list), "images should be a list."
            assert len(images) == input_ids.shape[0] == 1, "len(images) and input_ids.shape[0] should be 1."

            images = images[0]
            images = images.to(self.dtype)
            if self.dynamic_image_version == 'native_resolution':
                grid_hw = grid_hw[0]
                if image_seq_lens is not None:
                    image_seq_lens = image_seq_lens[0]
            else:
                grid_hw = None
                image_seq_lens = None
            
            assert grid_hw is not None
            # image gen
            pixel_values_updated, image_gen_z, image_gen_v, image_gen_t, image_gen_pos_ids, image_gen_noise_scale = self.prepare_image_gen_targets(images, image_for_gen_flags[0], grid_hw)

            vit_embeds, vit_embeds_image_gen, vit_moe_outputs, cls_embeds = self.extract_feature(     # NOTE:
                images, pixel_values_updated, image_flags=image_flags, return_cls=True, grid_hw=grid_hw
            )
            assert cls_embeds is None   # NOTE:

            if vit_moe_outputs is not None and len(vit_moe_outputs) > 0:
                moe_outputs = vit_moe_outputs
            if image_flags is not None and len(image_flags) > 0 and len(image_flags[0]) > 0 and grid_hw == None:       # NOTE:
                image_flags = image_flags[0]
                image_flags = image_flags.squeeze(-1)
                vit_embeds = vit_embeds[image_flags == 1]
                if (
                    image_con_flags is not None
                    and len(image_con_flags) > 0
                    and image_con_flags[0] is not None
                    and len(image_con_flags[0]) > 0
                ):
                    image_con_flags = image_con_flags[0].bool()
            elif image_flags is None:
                raise ValueError


        if hasattr(self, "language_model"):
            # no pp or first llm pipeline stage
            if (gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0) or (
                not gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
            ):
                if gpc.is_using_parallel_mode(ParallelMode.PIPELINE) and (
                    gpc.get_local_rank(ParallelMode.PIPELINE) - 1 == 0
                ):
                    vit_embeds = hidden_states if self.pure_llm is False else None

                # attention_mask: compute attention on the places where the value is 1
                assert hasattr(self.language_model, "tok_embeddings"), "The language model should have tok_embeddings!"
                if self.pure_llm is False:
                    selected = input_ids == self.img_context_token_id   # NOTE:
                    # for dummy dataset, the img_context_token_id is not in the vocab
                    if self.vocab_size < self.img_context_token_id:
                        input_ids = input_ids.clone()  # avoid inplace modification on ISP view
                        input_ids[selected] = 0
                hidden_states = self.language_model.tok_embeddings(input_ids)

                if pure_text and self.pure_llm is False:
                    hidden_states = hidden_states + 0 * vit_embeds.sum() + 0 * vit_embeds_image_gen.sum()

                    # NOTE:
                    assert indexes.ndim == 1, f"Expected 1D tensor for 'indexes', got shape {indexes.shape}"
                    zero_indexes = torch.zeros(indexes.shape[0], 2, dtype=indexes.dtype, device=indexes.device)
                    indexes =  torch.cat([indexes[:, None], zero_indexes], dim=-1)
                    # In ISP mode, indicators must match FULL sequence length
                    if self.tp_mode == "isp" and self.tp_size > 1:
                        full_seq_len = gather_forward_split_backward(input_ids.clone(), ParallelMode.TENSOR, dim=1).shape[1]
                        modality_indicators = torch.ones(full_seq_len, dtype=torch.long, device=input_ids.device) * -1
                        image_gen_indicators = torch.zeros(full_seq_len, dtype=torch.bool, device=input_ids.device).view(1, -1)
                        dup_boundary = torch.zeros(full_seq_len, dtype=torch.bool, device=input_ids.device)
                        # Gather hidden_states and indexes to full for downstream packing
                        hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=1)
                        indexes = gather_forward_split_backward(indexes.unsqueeze(0), ParallelMode.TENSOR, dim=1).squeeze(0)
                    else:
                        modality_indicators = torch.ones_like(input_ids[0]) * -1
                        image_gen_indicators = torch.zeros_like(input_ids[0], dtype=torch.bool).view(1, -1)
                        dup_boundary = torch.zeros_like(input_ids[0], dtype=torch.bool)
                    image_gen_t = None
                    image_gen_noise_scale = None
                    
                elif self.pure_llm is False:
                    vit_embeds = vit_embeds.reshape((-1, vit_embeds.shape[-1]))    # NOTE:
                    # try:
                    # the 'True' in selected might be distributed across different rank unevenly.
                    # For example, tp rank0: [True, False, False]; tp rank1: [True, True, True].
                    # Therefore, we need to split the vit_embeds according to the selected.
                    if self.tp_mode == "isp" and self.tp_size > 1:
                        # ISP mode: gather to full sequence, align context tokens per-image,
                        # then pack/split later for local LLM compute.
                        merge_size = round(1 / self.downsample_ratio)
                        vision_image_seq_lens = get_image_seq_lens(grid_hw, merge_size)
                        if image_seq_lens is None:
                            expected_image_seq_lens = vision_image_seq_lens
                        else:
                            if isinstance(image_seq_lens, torch.Tensor):
                                expected_image_seq_lens = [int(x) for x in image_seq_lens.tolist()]
                            else:
                                expected_image_seq_lens = [int(x) for x in image_seq_lens]
                            if expected_image_seq_lens != vision_image_seq_lens:
                                raise RuntimeError(
                                    "[image_gen][isp] image_seq_lens/grid_hw mismatch: "
                                    f"prompt_meta={expected_image_seq_lens} grid={vision_image_seq_lens}"
                                )

                        image_for_gen_flags_list = [bool(flag) for flag in image_for_gen_flags[0].tolist()]
                        gen_image_seq_lens = [
                            seq_len for seq_len, is_gen in zip(expected_image_seq_lens, image_for_gen_flags_list) if is_gen
                        ]

                        full_selected = gather_forward_split_backward(selected, ParallelMode.TENSOR, dim=1)
                        full_hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.TENSOR, dim=1)
                        full_indexes = gather_forward_split_backward(indexes.unsqueeze(0), ParallelMode.TENSOR, dim=1).squeeze(0)

                        if full_indexes.ndim != 1:
                            full_indexes = full_indexes.reshape(-1)

                        abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(grid_hw // merge_size, device=indexes.device)

                        modality_indicators, prompt_image_seq_lens = build_modality_indicators_from_context_runs(full_selected)
                        full_selected, _, kept_image_seq_lens = align_selected_to_image_seq_lens(
                            full_selected,
                            modality_indicators,
                            expected_image_seq_lens,
                        )
                        prompt_total_ctx = int(sum(prompt_image_seq_lens))
                        expected_total_ctx = int(sum(int(x) for x in expected_image_seq_lens))
                        kept_total_ctx = int(sum(int(x) for x in kept_image_seq_lens))
                        dropped_ctx = max(prompt_total_ctx - kept_total_ctx, 0)
                        if prompt_total_ctx != expected_total_ctx:
                            logger.warning(
                                "[image_gen][isp] image context token count mismatch: prompt=%s expected=%s kept=%s dropped=%s",
                                prompt_total_ctx,
                                expected_total_ctx,
                                kept_total_ctx,
                                dropped_ctx,
                            )

                        if prompt_image_seq_lens != expected_image_seq_lens:
                            mismatched_images, first_mismatch_idx, prompt_len, expected_len = summarize_image_seq_mismatch(
                                prompt_image_seq_lens,
                                expected_image_seq_lens,
                            )
                            logger.warning(
                                "[image_gen][isp] image context run mismatch: prompt=%s expected=%s kept=%s "
                                "prompt_runs=%s expected_images=%s mismatched_images=%s "
                                "first_mismatch_idx=%s prompt_len=%s expected_len=%s",
                                sum(prompt_image_seq_lens),
                                sum(expected_image_seq_lens),
                                sum(kept_image_seq_lens),
                                len(prompt_image_seq_lens),
                                len(expected_image_seq_lens),
                                mismatched_images,
                                first_mismatch_idx,
                                prompt_len,
                                expected_len,
                            )

                        vit_embeds = slice_tensor_by_image_lens(vit_embeds, expected_image_seq_lens, kept_image_seq_lens, dim=0)
                        vit_embeds_image_gen = slice_tensor_by_image_lens(vit_embeds_image_gen, expected_image_seq_lens, kept_image_seq_lens, dim=0)
                        abs_pos_h = slice_tensor_by_image_lens(abs_pos_h, expected_image_seq_lens, kept_image_seq_lens, dim=0)
                        abs_pos_w = slice_tensor_by_image_lens(abs_pos_w, expected_image_seq_lens, kept_image_seq_lens, dim=0)

                        raw_image_gen_indicators = self.build_image_gen_indicators(modality_indicators, image_for_gen_flags[0]).view(1, -1)
                        raw_dup_indicators = self.build_image_gen_indicators(modality_indicators, is_image_duplicated_for_und_flags[0]).view(1, -1)
                        image_gen_indicators = raw_image_gen_indicators & full_selected
                        image_duplicated_for_und_indicators = raw_dup_indicators & full_selected
                        dup_boundary = build_dup_boundary(
                            modality_indicators.view(-1),
                            image_gen_indicators.view(-1),
                            image_duplicated_for_und_indicators.view(-1),
                        )

                        kept_gen_image_seq_lens = [seq_len for seq_len, is_gen in zip(kept_image_seq_lens, image_for_gen_flags_list) if is_gen]

                        if image_gen_z is not None:
                            image_gen_z = slice_tensor_by_image_lens(image_gen_z, gen_image_seq_lens, kept_gen_image_seq_lens, dim=0)
                            image_gen_v = slice_tensor_by_image_lens(image_gen_v, gen_image_seq_lens, kept_gen_image_seq_lens, dim=0)
                            image_gen_t = slice_tensor_by_image_lens(image_gen_t, gen_image_seq_lens, kept_gen_image_seq_lens, dim=0)
                            image_gen_noise_scale = slice_tensor_by_image_lens(
                                image_gen_noise_scale, gen_image_seq_lens, kept_gen_image_seq_lens, dim=0
                            )
                            image_gen_pos_ids = slice_tensor_by_image_lens(image_gen_pos_ids, gen_image_seq_lens, kept_gen_image_seq_lens, dim=0)

                            k = int(image_gen_indicators.sum().item())
                            m = image_gen_z.shape[0]
                            if m < k:
                                logger.warning(
                                    "[image_gen][isp] indicators mismatch: indicators=%s > targets=%s.",
                                    k,
                                    m,
                                )
                                flat_indicators = image_gen_indicators.view(-1)
                                true_indices = torch.nonzero(flat_indicators).squeeze(-1)
                                flat_indicators[true_indices[m:]] = False
                                image_gen_indicators = flat_indicators.view(image_gen_indicators.shape)
                                k = m
                            elif k != m:
                                logger.warning(
                                    "[image_gen][isp] indicators mismatch: indicators=%s < targets=%s.",
                                    k,
                                    m,
                                )
                            image_gen_z = image_gen_z[:k]
                            image_gen_v = image_gen_v[:k]
                            image_gen_t = image_gen_t[:k]
                            image_gen_noise_scale = image_gen_noise_scale[:k]
                            image_gen_pos_ids = image_gen_pos_ids[:k]

                        full_hidden_states = full_hidden_states.clone()
                        gen_flags = image_gen_indicators[full_selected]
                        mixed = torch.where(gen_flags[:, None], vit_embeds_image_gen, vit_embeds)
                        full_hidden_states[full_selected] = mixed

                        pos_h = torch.zeros_like(full_indexes)
                        pos_w = torch.zeros_like(full_indexes)
                        pos_h[full_selected[0]] = abs_pos_h.to(dtype=full_indexes.dtype)
                        pos_w[full_selected[0]] = abs_pos_w.to(dtype=full_indexes.dtype)
                        indexes = torch.stack([full_indexes, pos_h, pos_w], dim=-1)
                        hidden_states = full_hidden_states
                    else:
                        # NOTE:
                        abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(grid_hw // int(1 / self.downsample_ratio), device=indexes.device)
                        pos_h = torch.zeros_like(indexes)
                        pos_w = torch.zeros_like(indexes)
                        pos_h[selected[0]] = abs_pos_h.to(dtype=indexes.dtype)
                        pos_w[selected[0]] = abs_pos_w.to(dtype=indexes.dtype)
                        indexes = torch.stack([indexes, pos_h, pos_w], dim=-1)

                        img_start_flags = (input_ids[0] == self.img_start_token_id).long()
                        shifted_flags = torch.cat([torch.zeros(1, dtype=torch.long, device=input_ids.device), img_start_flags], dim=0)[:-1]
                        modality_indicators = shifted_flags.cumsum(0)
                        modality_indicators[input_ids[0] != self.img_context_token_id] = -1
                        image_gen_indicators = self.build_image_gen_indicators(modality_indicators, image_for_gen_flags[0]).view(1, -1)
                        image_duplicated_for_und_indicators = self.build_image_gen_indicators(modality_indicators, is_image_duplicated_for_und_flags[0]).view(1, -1)
                        dup_boundary = build_dup_boundary(modality_indicators.view(-1), image_gen_indicators.view(-1), image_duplicated_for_und_indicators.view(-1))

                        if image_gen_z is not None:
                            # avoid length mismatch in some rare cases
                            k = int(image_gen_indicators.sum().item())
                            m = image_gen_z.shape[0]
                            if m < k:
                                logger.warning(f"[image_gen] indicators length mismatch: indicators={k} > targets={m}. Truncating indicators to match targets.")
                                flat_indicators = image_gen_indicators.view(-1)
                                true_indices = torch.nonzero(flat_indicators).squeeze(-1)
                                indices_to_mask = true_indices[m:]
                                flat_indicators[indices_to_mask] = False
                                image_gen_indicators = flat_indicators.view(image_gen_indicators.shape)
                                k = m
                            elif k != m:
                                logger.warning(f"[image_gen] indicators length mismatch: indicators={k} < targets={m}. Truncating targets to match indicators.")
                            image_gen_z = image_gen_z[:k]
                            image_gen_v = image_gen_v[:k]
                            image_gen_t = image_gen_t[:k]
                            image_gen_noise_scale = image_gen_noise_scale[:k]
                            image_gen_pos_ids = image_gen_pos_ids[:k]
              
                        hidden_states = hidden_states.clone()
                        gen_flags = image_gen_indicators[selected]
                        mixed = torch.where(gen_flags[:, None], vit_embeds_image_gen, vit_embeds)
                        hidden_states[selected] = mixed

                if image_gen_indicators.sum() == 0:
                    hidden_states = hidden_states + 0 * vit_embeds_image_gen.sum()

                if image_gen_t is not None:
                    if pad_dummy_image_gen:
                        pad_image_gen_t = torch.zeros(
                            (pad_dummy_image_num,),
                            device=image_gen_t.device, dtype=image_gen_t.dtype
                        )
                        image_gen_t = torch.cat([pad_image_gen_t, image_gen_t])
                        pad_image_gen_noise_scale = torch.zeros(
                            (pad_dummy_image_num,),
                            device=image_gen_noise_scale.device, dtype=image_gen_noise_scale.dtype
                        )
                        image_gen_noise_scale = torch.cat([pad_image_gen_noise_scale, image_gen_noise_scale])
                        pad_image_gen_z = torch.zeros(
                            (pad_dummy_image_num, image_gen_z.shape[1]),
                            device=image_gen_z.device, dtype=image_gen_z.dtype
                        )
                        image_gen_z = torch.cat([pad_image_gen_z, image_gen_z])
                        pad_image_gen_v = torch.zeros(
                            (pad_dummy_image_num, image_gen_v.shape[1]),
                            device=image_gen_v.device, dtype=image_gen_v.dtype
                        )
                        image_gen_v = torch.cat([pad_image_gen_v, image_gen_v])
                    timestep_embeddings = self.fm_modules['timestep_embedder'](image_gen_t)
                    if self.add_noise_scale_embedding:
                        noise_scale_embedding = self.fm_modules['noise_scale_embedder'](image_gen_noise_scale)
                        timestep_embeddings = timestep_embeddings + noise_scale_embedding
                    if pad_dummy_image_gen:
                        hidden_states[image_gen_indicators] += timestep_embeddings[pad_dummy_image_num:]
                    else:
                        hidden_states[image_gen_indicators] += timestep_embeddings
                else:
                    if pad_dummy_image_gen:
                        patch_size = self.config.vision_config.patch_size
                        merge_size = round(1 / self.downsample_ratio) 
                        patch_size_after_downsample = patch_size * merge_size
                        image_gen_t = torch.zeros(
                            (pad_dummy_image_num,),
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )
                        image_gen_noise_scale = torch.zeros(
                            (pad_dummy_image_num,),
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )
                        timestep_embeddings = self.fm_modules['timestep_embedder'](image_gen_t)
                        if self.add_noise_scale_embedding:
                            noise_scale_embedding = self.fm_modules['noise_scale_embedder'](image_gen_noise_scale)
                            timestep_embeddings = timestep_embeddings + noise_scale_embedding
                        image_gen_z = torch.zeros(
                            (pad_dummy_image_num, patch_size_after_downsample**2*3),
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )
                        image_gen_v = torch.zeros(
                            (pad_dummy_image_num, patch_size_after_downsample**2*3),
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )                 

            if self.language_model.embed_grad_scale != 1:
                hidden_states, hidden_states_before_output = (
                    self.language_model.embed_grad_scale * hidden_states
                    + (1 - self.language_model.embed_grad_scale) * hidden_states.detach()
                )
            
            # NOTE:
            if self.tp_mode == "isp" and self.tp_size > 1:
                _full_input_ids = gather_forward_split_backward(input_ids.clone(), ParallelMode.TENSOR, dim=1)
                _has_pad = _full_input_ids[0][-1] == 0
            else:
                _has_pad = input_ids[0][-1] == 0
            document_ids = _offsets_to_doc_ids_tensor(cu_seqlens, has_pad=_has_pad)
            # for the mot strucutre, we split the vision gen tokens and other tokens and pack them
            hidden_states_packed, indexes_packed, document_ids_packed, modality_indicators_packed, image_gen_indicators_packed, perm, inv_perm, token_pos_packed, dup_boundary_packed = pack_two_branch_sequence(
                hidden_states=hidden_states,
                indexes=indexes,
                document_ids=document_ids,
                modality_indicators=modality_indicators,
                image_gen_indicators=image_gen_indicators,
                dup_boundary=dup_boundary,
            )

            if pad_dummy_image_gen:
                device = hidden_states_packed.device
                dtype = hidden_states_packed.dtype

                pad_h = torch.zeros(
                    (hidden_states_packed.shape[0], pad_dummy_image_num, hidden_states_packed.shape[2]),
                    device=device, dtype=dtype
                )
                pad_h += timestep_embeddings[:pad_dummy_image_num].unsqueeze(0)
                hidden_states_packed = torch.cat([pad_h, hidden_states_packed], dim=1)

                pad_idx = torch.zeros(
                    (pad_dummy_image_num, indexes_packed.shape[-1]),
                    device=indexes_packed.device,
                    dtype=indexes_packed.dtype
                )
                indexes_packed = torch.cat([pad_idx, indexes_packed], dim=0)

                new_doc = document_ids_packed.max() + 1
                pad_doc = torch.full(
                    (pad_dummy_image_num,),
                    new_doc,
                    device=document_ids_packed.device,
                    dtype=document_ids_packed.dtype
                )
                document_ids_packed = torch.cat([pad_doc, document_ids_packed], dim=0)

                pad_mod = torch.full(
                    (pad_dummy_image_num,),
                    -1,
                    device=modality_indicators_packed.device,
                    dtype=modality_indicators_packed.dtype
                )
                modality_indicators_packed = torch.cat([pad_mod, modality_indicators_packed], dim=0)

                pad_gen = torch.ones(
                    (pad_dummy_image_num, ),
                    device=image_gen_indicators_packed.device,
                    dtype=image_gen_indicators_packed.dtype
                )
                image_gen_indicators_packed = torch.cat(
                    [pad_gen, image_gen_indicators_packed],
                    dim=0
                )

                pad_dup_boundary = torch.zeros(
                    (pad_dummy_image_num, ),
                    device=dup_boundary_packed.device,
                    dtype=dup_boundary_packed.dtype
                )
                dup_boundary_packed = torch.cat(
                    [pad_dup_boundary, dup_boundary_packed],
                    dim=0
                )

                pad_pos = torch.zeros(
                    (pad_dummy_image_num,),
                    device=token_pos_packed.device,
                    dtype=token_pos_packed.dtype
                )
                token_pos_packed = torch.cat([pad_pos, token_pos_packed], dim=0)

            flex_mask, padlen = create_flex_mask_padding_image_gen(document_ids_packed, modality_indicators=modality_indicators_packed, image_gen_indicators=image_gen_indicators_packed.view(-1), token_pos=token_pos_packed, dup_boundary=dup_boundary_packed, div_num=128, compile_mask=True)

            def _build_cu_seqlens_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
                if doc_ids.numel() == 0:
                    return torch.zeros(1, dtype=torch.int32, device=doc_ids.device)
                starts = torch.ones_like(doc_ids, dtype=torch.bool)
                starts[1:] = doc_ids[1:] != doc_ids[:-1]
                seg_ids = torch.cumsum(starts.to(torch.int32), dim=0) - 1
                num_segs = int(seg_ids[-1].item()) + 1
                counts = torch.bincount(seg_ids, minlength=num_segs).to(torch.int32)
                return torch.cat(
                    [torch.zeros(1, dtype=torch.int32, device=counts.device), counts.cumsum(0)],
                    dim=0,
                )

            if self.tp_mode == "isp" and self.tp_size > 1:
                input_ids_packed = _full_input_ids[:, perm]
            else:
                input_ids_packed = input_ids[:, perm]
            if pad_dummy_image_gen:
                pad_input_ids = torch.zeros(
                    (input_ids_packed.shape[0], pad_dummy_image_num),
                    device=input_ids_packed.device,
                    dtype=input_ids_packed.dtype,
                )
                input_ids_packed = torch.cat([pad_input_ids, input_ids_packed], dim=1)

            gen_mask = image_gen_indicators_packed.bool()
            und_mask = ~gen_mask
            gen_input_ids = input_ids_packed[:, gen_mask]
            und_input_ids = input_ids_packed[:, und_mask]
            gen_cu_seqlens = _build_cu_seqlens_from_doc_ids(document_ids_packed[gen_mask])
            und_cu_seqlens = _build_cu_seqlens_from_doc_ids(document_ids_packed[und_mask])
            und_image_inds = (modality_indicators_packed[und_mask] != -1).view(1, -1)

            log_every = 50
            batch_count = int(getattr(gpc.config, "batch_count", 0))
            if gpc.is_rank_for_log() and batch_count % log_every == 0:
                logger.info(
                    "[moe-pack] step=%s total=%s gen=%s und=%s gen_cu_last=%s und_cu_last=%s und_img=%s",
                    batch_count,
                    input_ids_packed.shape[1],
                    gen_input_ids.shape[1],
                    und_input_ids.shape[1],
                    int(gen_cu_seqlens[-1].item()) if gen_cu_seqlens.numel() > 0 else 0,
                    int(und_cu_seqlens[-1].item()) if und_cu_seqlens.numel() > 0 else 0,
                    int(und_image_inds.sum().item()) if und_image_inds.numel() > 0 else 0,
                )
            # All-rank diagnostic: gather und stats across DP ranks (only on log steps)
            if batch_count % log_every == 0:
                _und_img_local = torch.tensor(
                    [int(und_image_inds.sum().item()) if und_image_inds.numel() > 0 else 0],
                    dtype=torch.long, device=input_ids_packed.device,
                )
                # count non-padding und tokens: segments where input_ids are not all-zero
                _und_cu = und_cu_seqlens
                _und_valid = 0
                for _i in range(len(_und_cu) - 1):
                    if not torch.all(und_input_ids[0][_und_cu[_i]:_und_cu[_i+1]] == 0):
                        _und_valid += (_und_cu[_i+1] - _und_cu[_i]).item()
                _und_valid_t = torch.tensor([_und_valid], dtype=torch.long, device=input_ids_packed.device)
                _und_total_t = torch.tensor([und_input_ids.shape[1]], dtype=torch.long, device=input_ids_packed.device)

                _world = torch.distributed.get_world_size()
                _local_stats = torch.stack([_und_img_local, _und_valid_t, _und_total_t]).view(3)  # [3]
                _all_stats = [torch.zeros(3, dtype=torch.long, device=input_ids_packed.device) for _ in range(_world)]
                torch.distributed.all_gather(_all_stats, _local_stats)
                if gpc.is_rank_for_log():
                    _imgs = [s[0].item() for s in _all_stats]
                    _valids = [s[1].item() for s in _all_stats]
                    _totals = [s[2].item() for s in _all_stats]
                    _zero_img_ranks = sum(1 for x in _imgs if x == 0)
                    logger.info(
                        "[moe-pack-allrank] step=%s und_img_per_rank=%s "
                        "zero_img_ranks=%s/%s "
                        "und_valid_tokens: sum=%s min=%s max=%s "
                        "und_total_tokens: sum=%s",
                        batch_count,
                        _imgs,
                        _zero_img_ranks, _world,
                        sum(_valids), min(_valids), max(_valids),
                        sum(_totals),
                    )

            flex_mask_for_llm = flex_mask
            padlen_for_llm = padlen
            if self.tp_mode == "isp" and self.tp_size > 1:
                packed_full_len = hidden_states_packed.shape[1]
                packed_div = packed_full_len // self.tp_size
                packed_mod = packed_full_len % self.tp_size
                packed_split_sizes = get_split_size(self.tp_size, packed_div, packed_mod)

                hidden_states_packed_for_llm = split_forward_gather_backward(
                    hidden_states_packed,
                    ParallelMode.TENSOR,
                    dim=1,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                )
                indexes_packed_for_llm = split_forward_gather_backward(
                    indexes_packed,
                    ParallelMode.TENSOR,
                    dim=0,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                )

                # image_gen_indicators must be local (LLM splits local hidden_states into gen/und)
                image_gen_indicators_packed_local = split_forward_gather_backward(
                    image_gen_indicators_packed.view(1, -1),
                    ParallelMode.TENSOR,
                    dim=1,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                ).view(-1)
                image_gen_indicators_for_llm = image_gen_indicators_packed_local.view(1, -1)

                # MoE gate_logits are LOCAL in ISP mode, so gen/und derived tensors
                # for load_balancing_loss must also be LOCAL
                local_gen_mask = image_gen_indicators_packed_local.bool()
                local_und_mask = ~local_gen_mask
                local_input_ids = split_forward_gather_backward(
                    input_ids_packed,
                    ParallelMode.TENSOR,
                    dim=1,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                )
                local_doc_ids = split_forward_gather_backward(
                    document_ids_packed,
                    ParallelMode.TENSOR,
                    dim=0,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                )
                local_modality = split_forward_gather_backward(
                    modality_indicators_packed,
                    ParallelMode.TENSOR,
                    dim=0,
                    div=packed_div,
                    mod=packed_mod,
                    split_size=packed_split_sizes,
                )

                gen_input_ids = local_input_ids[:, local_gen_mask]
                und_input_ids = local_input_ids[:, local_und_mask]
                gen_cu_seqlens = _build_cu_seqlens_from_doc_ids(local_doc_ids[local_gen_mask])
                und_cu_seqlens = _build_cu_seqlens_from_doc_ids(local_doc_ids[local_und_mask])
                und_image_inds = (local_modality[local_und_mask] != -1).view(1, -1)
            else:
                hidden_states_packed_for_llm = hidden_states_packed
                indexes_packed_for_llm = indexes_packed
                image_gen_indicators_for_llm = image_gen_indicators_packed.view(1, -1)

            llm_outputs = self.language_model(
                hidden_states=hidden_states_packed_for_llm,
                input_ids=input_ids, # NOTE:
                image_gen_indicators=image_gen_indicators_for_llm,
                cu_seqlens=cu_seqlens, # NOTE:
                indexes=indexes_packed_for_llm,
                max_seqlen=kwargs["max_seqlen"],
                flex_mask=flex_mask_for_llm,    # NOTE:
                padlen=padlen_for_llm,    # NOTE:
                inference_params=inference_params,
                gen_image_inds=None,
                und_image_inds=und_image_inds,
                gen_input_ids=gen_input_ids,
                und_input_ids=und_input_ids,
                gen_cu_seqlens=gen_cu_seqlens,
                und_cu_seqlens=und_cu_seqlens,
            )

            llm_moe_outputs = []

            if len(llm_outputs) == 2:
                hidden_states, hidden_states_before_output = llm_outputs
            elif len(llm_outputs) == 3:
                hidden_states, hidden_states_before_output, llm_moe_outputs = llm_outputs
            elif len(llm_outputs) == 4 and \
                    (gpc.config.model_type.upper().startswith("DEEPSEEK3") or \
                     gpc.config.model_type.upper().startswith("QWEN3_MOE")):
                hidden_states_packed, hidden_states_before_output_packed, mtp_outputs, llm_moe_outputs = llm_outputs  # NOTE:
            else:
                # TODO HARD CODE FOR QWEN2
                hidden_states_packed = llm_outputs
                hidden_states_before_output_packed = None

            if len(llm_moe_outputs) > 0:
                moe_outputs = llm_moe_outputs

            # In ISP mode, LLM output is local (ISP-split). Gather back to full
            # before unpacking (which requires full-sequence perm/inv_perm).
            if self.tp_mode == "isp" and self.tp_size > 1:
                packed_local_len = packed_split_sizes[gpc.get_local_rank(ParallelMode.TENSOR)]
                if hidden_states_packed.shape[1] == packed_local_len:
                    hidden_states_packed = gather_forward_split_backward(
                        hidden_states_packed,
                        ParallelMode.TENSOR,
                        dim=1,
                        div=packed_div,
                        mod=packed_mod,
                        split_size=packed_split_sizes,
                    )
                if hidden_states_before_output_packed is not None and hidden_states_before_output_packed.shape[1] == packed_local_len:
                    hidden_states_before_output_packed = gather_forward_split_backward(
                        hidden_states_before_output_packed,
                        ParallelMode.TENSOR,
                        dim=1,
                        div=packed_div,
                        mod=packed_mod,
                        split_size=packed_split_sizes,
                    )

            if pad_dummy_image_gen:
                hidden_states = unpack_two_branch_sequence(hidden_states_packed[:, pad_dummy_image_num:], inv_perm)
                if hidden_states_before_output_packed is not None:
                    hidden_states_before_output = torch.cat([hidden_states_before_output_packed[:, :pad_dummy_image_num], unpack_two_branch_sequence(hidden_states_before_output_packed[:, pad_dummy_image_num:], inv_perm)], 1)
            else:
                hidden_states = unpack_two_branch_sequence(hidden_states_packed, inv_perm)
                if hidden_states_before_output_packed is not None:
                    hidden_states_before_output = unpack_two_branch_sequence(hidden_states_before_output_packed, inv_perm)
     
        boi_loss = None

        # calculate image gen related loss
        if self.image_gen_loss_weight > 0:
            type_ids = type_ids.view(-1)
            image_gen_type_ids = type_ids[image_gen_indicators.view(-1)]
            if pad_dummy_image_gen:
                image_gen_indicators = torch.cat([image_gen_indicators_packed[:pad_dummy_image_num].view(1, -1), image_gen_indicators], 1)
            image_gen_hidden_states = hidden_states_before_output.view(-1, hidden_states_before_output.shape[-1])[image_gen_indicators.view(-1)]

            image_gen_pred_x = self.fm_modules['fm_head'](image_gen_hidden_states)

            # calculate v loss
            image_gen_pred_v = (image_gen_pred_x - image_gen_z) / (1 - image_gen_t.view(-1, 1)).clamp_min(self.t_eps)
            image_gen_loss = F.mse_loss(image_gen_pred_v, image_gen_v, reduction='none') * self.image_gen_loss_weight
            if pad_dummy_image_gen:
                image_gen_loss[:pad_dummy_image_num] = image_gen_loss[:pad_dummy_image_num] * 0
            else:
                pad_dummy_image_num = 0
            image_gen_loss = image_gen_loss.mean(dim=1)

            image_gen_loss_t2i_indicators = (image_gen_type_ids==3)
            image_gen_loss_t2i = global_all_reduce_loss(image_gen_loss[pad_dummy_image_num:][image_gen_loss_t2i_indicators].sum(), image_gen_loss_t2i_indicators.sum())

            image_gen_loss_editing_indicators = (image_gen_type_ids==4)
            image_gen_loss_editing = global_all_reduce_loss(image_gen_loss[pad_dummy_image_num:][image_gen_loss_editing_indicators].sum(), image_gen_loss_editing_indicators.sum())

            image_gen_loss_interleave_indicators = (image_gen_type_ids==5)
            image_gen_loss_interleave = global_all_reduce_loss(image_gen_loss[pad_dummy_image_num:][image_gen_loss_interleave_indicators].sum(), image_gen_loss_interleave_indicators.sum())

            losses_for_log_only = {}
            losses_for_log_only["image_gen_loss_t2i"] = image_gen_loss_t2i
            losses_for_log_only["image_gen_loss_editing"] = image_gen_loss_editing
            losses_for_log_only["image_gen_loss_interleave"] = image_gen_loss_interleave

            image_gen_loss_weight = []
            for image_i in range(len(image_for_gen_flags[0])):
                if image_for_gen_flags[0][image_i]:
                    cur_image_h = grid_hw[image_i, 0]
                    cur_image_w = grid_hw[image_i, 1]
                    cur_image_token_num = cur_image_h * cur_image_w
                    merge_size = round(1 / self.downsample_ratio) 
                    cur_image_seq_len = cur_image_token_num // (merge_size**2)
                    image_gen_loss_weight.extend([(1 / cur_image_seq_len**0.5)]*cur_image_seq_len)
            image_gen_loss_weight = torch.tensor(image_gen_loss_weight, dtype=torch.float32, device=image_gen_loss.device)
            image_gen_loss[pad_dummy_image_num:] *= image_gen_loss_weight
            image_gen_loss_weight_sum = image_gen_loss_weight.sum()
            dist.all_reduce(image_gen_loss_weight_sum, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
            image_gen_loss = image_gen_loss.sum() / image_gen_loss_weight_sum.clamp_min(1)



        else:
            image_gen_loss = None
            losses_for_log_only = None

        if len(moe_outputs) > 0 and moe_outputs[0] is not None:
            for moe_output in moe_outputs:
                if moe_output.moe_loss is not None:
                    moe_losses.append(moe_output.moe_loss)
                if moe_output.z_loss is not None:
                    moe_z_losses.append(moe_output.z_loss)
                if moe_output.routed_coef_loss is not None:
                    moe_coef_losses.append(moe_output.routed_coef_loss)

                # moe monitor
                moe_monitor_cfg = gpc.config.moe_monitor
                if gpc.config.batch_count % moe_monitor_cfg.get("interval_steps", 10) == 0:
                    if moe_output.moe_loss is not None and moe_monitor_cfg.get("layer_moe_loss", False):
                        gpc.metric["moe_loss"].append(moe_output.moe_loss)
                    if moe_output.z_loss is not None and moe_monitor_cfg.get("layer_z_loss", False):
                        gpc.metric["moe_z_loss"].append(moe_output.z_loss)
                    if moe_output.routed_coef_loss is not None and moe_monitor_cfg.get("layer_coef_loss", False):
                        gpc.metric["moe_coef_loss"].append(moe_output.routed_coef_loss)
                    if moe_output.routed_coef is not None and moe_monitor_cfg.get("route_coef", False):
                        gpc.metric["moe_route_coef"].append(moe_output.routed_coef)
                    if moe_output.gates_max is not None:
                        gpc.metric["moe_gates_max"].append(moe_output.gates_max)
                    if moe_output.drop_ratio is not None:
                        gpc.metric["moe_drop_ratio"].append(moe_output.drop_ratio)

            moe_coef = getattr(gpc.config.loss, "expert_balance_coef", 0.0)
            moe_z_coef = getattr(gpc.config.loss, "moe_z_loss_coef", 0.0)
            expert_coef_balance_coef = getattr(gpc.config.loss, "expert_coef_balance_coef", 0.0)
            if len(moe_losses) > 0:
                moe_losses = [sum(moe_losses) * moe_coef]
            if len(moe_z_losses) > 0:
                moe_z_losses = [sum(moe_z_losses) * moe_z_coef]
            if len(moe_coef_losses) > 0:
                moe_coef_losses = [sum(moe_coef_losses) * expert_coef_balance_coef]

        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            if hasattr(self, "vision_model"):
                return vit_embeds, boi_loss, moe_losses, moe_z_losses, moe_coef_losses, image_gen_loss, losses_for_log_only
            return hidden_states, mtp_outputs, boi_loss, moe_losses, moe_z_losses, moe_coef_losses, image_gen_loss, losses_for_log_only
        else:
            return hidden_states, mtp_outputs, boi_loss, moe_losses, moe_z_losses, moe_coef_losses, image_gen_loss, losses_for_log_only

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == "v1":
            if gpc.is_rank_for_log():
                warnings.warn(
                    "In ps_version 'v1', the height and width have not been swapped back, "
                    "which results in a transposed image."
                )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, pixel_values_noised, image_flags=None, return_cls=False, grid_hw=None):
        

        if image_flags is not None and isinstance(image_flags, list):
            assert len(image_flags) == 1
            image_flags = image_flags[0]

        assert self.select_layer == -1
        if self.select_layer == -1:
            outputs = self.vision_model(
                pixel_values=pixel_values, image_flags=image_flags, output_hidden_states=False, return_dict=True, grid_hw=grid_hw
            )
            outputs_image_gen = self.fm_modules['vision_model_mot_gen'](
                pixel_values=pixel_values_noised, image_flags=image_flags, output_hidden_states=False, return_dict=True, grid_hw=grid_hw
            )
            vit_embeds = outputs.last_hidden_state
            vit_embeds_image_gen = outputs_image_gen.last_hidden_state
        else:
            outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True, grid_hw=grid_hw)
            vit_embeds = outputs.hidden_states[self.select_layer]

        moe_outputs = outputs.moe_outputs
        seq_len = outputs.sequence_length
        
        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            # all gather the sequence to conduct pixel shuffle
            div_seq = seq_len // self.tp_size
            mod_seq = seq_len % self.tp_size
            seq_split_size = get_split_size(self.tp_size, div_seq, mod_seq)
            vit_embeds = gather_forward_split_backward(
                vit_embeds,
                ParallelMode.TENSOR,
                dim=-2,
                div=div_seq,
                mod=mod_seq,
                split_size=seq_split_size,
            )

        cls_embeds = None
        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            vit_embeds = split_forward_gather_backward(vit_embeds, ParallelMode.TENSOR, dim=-2)

        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            vit_embeds = gather_forward_split_backward(vit_embeds, ParallelMode.TENSOR, dim=-2)

        if return_cls:
            return vit_embeds, vit_embeds_image_gen, moe_outputs, cls_embeds
        else:
            return vit_embeds, vit_embeds_image_gen, moe_outputs

def build_pipeline_partition_mot_model(**kwargs):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. sensenovalm_accelerator.device() by default.

    """

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        kwargs["first"] = True
        kwargs["last"] = True
        return SenseNovaVLChatMoTModel(**kwargs)

    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    kwargs["first"] = False
    kwargs["last"] = False

    if pipeline_rank == 0:
        kwargs["first"] = True
    if pipeline_rank == pipeline_size - 1:
        kwargs["last"] = True

    return SenseNovaVLChatMoTModel(**kwargs)
