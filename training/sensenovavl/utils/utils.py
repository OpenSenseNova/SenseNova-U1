# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# RoPE math (1D / 2D) follows RoFormer (Su et al., 2021).
# References:
#   https://arxiv.org/abs/2104.09864
#   https://github.com/ZhuiyiTechnology/roformer
import torch
import torch.distributed as dist
from PIL import Image, ImageFile, PngImagePlugin

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device

def precompute_rope_freqs_sincos(
    dim: int, max_position: int, base: float = 10000.0, device=None
):
    """预计算 RoPE 的 cos 和 sin 值 (1D)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
):
    """对输入张量的一部分应用1D RoPE。"""
    
    cos = cos_cached[positions] # Shape: (positions.shape, dim_part / 2)
    sin = sin_cached[positions] # Shape: (positions.shape, dim_part / 2)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor
):
    """应用2D RoPE到输入张量x。"""
    dim = x.shape[-1]
    dim_half = dim // 2

    # 假设我们将embedding的前半部分用于一个方向的RoPE，后半部分用于另一个方向
    # 例如，前一半给X坐标，后一半给Y坐标 (或者反过来，但要保持一致)
    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    # 将与 abs_positions_x 相关的旋转应用于 x_part_1
    rotated_part_1 = apply_rotary_emb_1d(
        x_part_1, cos_cached_x, sin_cached_x, abs_positions_x
    )
    # 将与 abs_positions_y 相关的旋转应用于 x_part_2
    rotated_part_2 = apply_rotary_emb_1d(
        x_part_2, cos_cached_y, sin_cached_y, abs_positions_y
    )
    
    # 将它们重新拼接起来。确保顺序与你分割时一致。
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


def init_pil():
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    MaximumDecompressedSize = 1024
    MegaByte = 2**20
    PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def load_safetensors(filename):
    from safetensors import safe_open

    model = safe_open(filename, framework="pt")
    state_dict = {}
    for k in model.keys():
        state_dict[k] = model.get_tensor(k)

    return state_dict


def check_image_fn(data, label):
    if gpc.config.model.use_moe is False or gpc.config.model.moe_location == "llm":
        return data, label

    images = data["images"]
    device = get_current_device()

    for image in images:
        shape = image.shape[0] if image is not None else 0
        shape_tensor = torch.tensor(shape).to(device)

        all_shapes = [torch.zeros_like(shape_tensor) for _ in range(gpc.get_world_size(ParallelMode.EXPERT))]
        dist.all_gather(all_shapes, shape_tensor, group=gpc.get_group(ParallelMode.EXPERT))

        assert all_shapes.count(all_shapes[0]) == len(
            all_shapes
        ), f"the number of image should be equal.{all_shapes.count(all_shapes[0])=}, {len(all_shapes)=}, {all_shapes=}."

    return data, label
