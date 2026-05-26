# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from typing import List
import torch

def repeat_all_but_last(s, token="<image>"):
    parts = s.split(token)
    if len(parts) <= 2:  # 0 or 1 occurrence
        return s
    return (token * 2).join(parts[:-1]) + token + parts[-1]

### For T2I and Editing

def find_subsequence_positions(seq: torch.Tensor, pattern: torch.Tensor):
    """
    seq: [L]
    pattern: [K]
    returns: positions where pattern starts
    """
    L, K = seq.numel(), pattern.numel()
    if L < K:
        return torch.empty(0, dtype=torch.long, device=seq.device)

    windows = seq.unfold(0, K, 1)  # [L-K+1, K]
    matches = (windows == pattern).all(dim=1)

    return matches.nonzero(as_tuple=True)[0].tolist()


def build_text_drop_mask_ranges_singleturn(
    user_content_start: int,
    user_content_end: int,
    image_start_positions: list,
    image_end_positions: list,
):
    """
    Build mask ranges inside [user_content_start, user_content_end),
    excluding all tokens that lie inside any image span
    [image_start_token, image_end_token] (inclusive).

    Returns a flattened list of ranges:
        [s1, e1, s2, e2, ...]
    where each range is [start, end) (end not included).

    If nothing needs to be masked, returns [-1, -1].

    Notes:
    - Image spans are clipped to the user interval
    - Image start/end tokens themselves are considered part of the image span
    """

    u0 = user_content_start
    u1 = user_content_end

    # Empty interval
    if u0 >= u1:
        return [-1, -1]

    # -------------------------------------------------
    # Step 1: Build image spans within user interval
    # Convert inclusive [start, end] → half-open [start, end+1)
    # -------------------------------------------------
    spans = []
    for s, e in zip(image_start_positions, image_end_positions):
        if e < s:
            continue

        s2 = max(s, u0)
        e2 = min(e + 1, u1)

        if s2 < e2:
            spans.append((s2, e2))

    # If there are no image spans, mask the whole interval
    if not spans:
        return [u0, u1]

    # -------------------------------------------------
    # Step 2: Merge overlapping or adjacent spans
    # -------------------------------------------------
    spans.sort()
    merged = []

    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:  # overlap or adjacent
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))

    # -------------------------------------------------
    # Step 3: Compute complement = regions to mask
    # -------------------------------------------------
    mask_ranges = []
    prev = u0

    for s, e in merged:
        if prev < s:
            mask_ranges.extend([prev, s])
        prev = max(prev, e)

    if prev < u1:
        mask_ranges.extend([prev, u1])

    return mask_ranges if mask_ranges else [-1, -1]



#### For interleave

def mask_all_prev_images(image_i: int, image_start_token_positions: List[int], image_end_token_positions: List[int]):
    """
    Mask ALL images strictly before current image_i: flatten [s, e+1] for j in [0, image_i).
    """
    out = []
    for j in range(image_i):
        s = image_start_token_positions[j]
        e = image_end_token_positions[j]
        if e >= s:
            out.extend([s, e + 1])  # inclusive -> half-open
    return out

def mask_text_for_current_image(image_i: int, user_pos: list, asst_pos: list, image_start_token_positions: List[int], image_end_token_positions: List[int]):
    """
    Build mask ranges for the 'text' drop case described by you:
    - only consider current round (cur_user + cur_asst)
    - search backward from current image in assistant prefix:
        skip consecutive images, then take nearest text segment until previous image boundary
    - if no image boundary encountered in assistant, or assistant has no text, include all user text
      (excluding user images)
    Returns flattened [s1,e1,s2,e2,...] (half-open).
    """
    cur_img_s = image_start_token_positions[image_i]

    # Current round: last ASSISTANT before current image, and last USER before that
    cur_asst = max(p for p in asst_pos if p < cur_img_s)
    cur_user = max(p for p in user_pos if p < cur_asst)

    # Exclude role tokens
    user_l = cur_user + 3
    user_r = cur_asst - 2
    asst_l = cur_asst + 3
    asst_r = cur_img_s

    # previous image index (may be -1)
    j = image_i - 1

    # 1) Skip consecutive images immediately before asst_r
    cursor = asst_r
    while j >= 0 and image_end_token_positions[j] >= asst_l and image_end_token_positions[j] == cursor - 1:
        cursor = image_start_token_positions[j]
        j -= 1
    text_end = cursor

    # 2) Stop text at the nearest previous image inside assistant content; else reach asst_l
    while j >= 0 and image_end_token_positions[j] < asst_l:
        j -= 1
    text_start = asst_l if j < 0 else max(asst_l, image_end_token_positions[j] + 1)

    asst_text = [text_start, text_end] if text_start < text_end else []

    # 3) Decide whether to also include USER text (excluding USER images)
    include_user = (text_start == asst_l) or (not asst_text)

    out = []

    # Mask USER text ranges in [user_l, user_r), excluding user images
    if include_user and user_l < user_r:
        prev = user_l
        for s, e in zip(image_start_token_positions, image_end_token_positions):
            if e < user_l:
                continue
            if s >= user_r:
                break
            s2 = max(s, user_l)
            e2 = min(e, user_r - 1)  # inclusive
            if prev < s2:
                out.extend([prev, s2])
            prev = max(prev, e2 + 1)  # inclusive -> next index
        if prev < user_r:
            out.extend([prev, user_r])

    # Append assistant text segment
    if asst_text:
        out.extend(asst_text)

    return out