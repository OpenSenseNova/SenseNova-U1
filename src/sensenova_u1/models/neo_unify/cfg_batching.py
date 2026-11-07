"""Batch CFG projections/MLPs while keeping each branch's prefix cache intact."""

import torch


class BatchedCFGCache:
    """Read-only prefix-cache view for generation with ``update_cache=False``.

    Only the preallocated image-token slots are overwritten. Attention runs
    per branch so unequal prefix lengths need neither padding nor dense masks.
    The surrounding projections and MLPs still run once on the combined batch.
    """

    def __init__(self, branches):
        if len(branches) < 2:
            raise ValueError("Batched CFG requires at least two branches")
        self.branches = tuple(branches)
        self.enabled = True
        for cache in self.branches:
            if len(cache.layers) != len(self.branches[0].layers) or any(
                getattr(layer, "flash_k_cache", None) is None or getattr(layer, "flash_v_cache", None) is None
                for layer in cache.layers
            ):
                raise ValueError("Batched CFG requires prepared prefix caches with matching layer counts")

    def get_seq_length(self, layer_idx=0):
        # Used only to construct the decoder's unused cache_position metadata;
        # actual RoPE indices and attention lengths stay specific to each branch.
        return self.branches[0].get_seq_length(layer_idx)

    def attention(self, layer_idx, q, k, v, attend, **kwargs):
        if q.shape[0] % len(self.branches):
            raise ValueError("CFG batch must divide evenly between branches")
        outputs = []
        for cache, query, key, value in zip(
            self.branches, q.chunk(len(self.branches)), k.chunk(len(self.branches)), v.chunk(len(self.branches))
        ):
            layer = cache.layers[layer_idx]
            start, end = layer.flash_prefix_len, layer.flash_prefix_len + key.shape[1]
            layer.flash_k_cache[:, start:end].copy_(key)
            layer.flash_v_cache[:, start:end].copy_(value)
            outputs.append(attend(query, layer.flash_k_cache[:, :end], layer.flash_v_cache[:, :end], **kwargs))
        return torch.cat(outputs, dim=0)


def predict_cfg_batched(predict_v, input_embeds, indexes, cache, t, z, image_token_num, **kwargs):
    """Keep temporary batched tensors out of the OOM recovery frame."""
    batch_size = input_embeds.shape[0]
    batched_indexes = torch.cat([index[:, None].expand(-1, batch_size, -1) for index in indexes], dim=1)
    if kwargs.get("timestep_embeddings") is not None:
        embeddings = kwargs["timestep_embeddings"]
        kwargs["timestep_embeddings"] = torch.cat([embeddings, embeddings])
    return predict_v(
        torch.cat([input_embeds, input_embeds]),
        batched_indexes,
        {"full_attention": None},
        cache,
        t,
        torch.cat([z, z]),
        image_token_num,
        **kwargs,
    ).chunk(2)


def cfg_batch_has_headroom(input_embeds, config, activation_budget=None):
    """Conservative workspace guard; an OOM retry remains the final fallback.

    Include residuals, QKV/norm temporaries, and gated MLP activations. For MoE,
    account for every selected expert's intermediate activations. Prefix KV
    storage is shared with the serial path and does not need to be duplicated.
    """
    if input_embeds.device.type not in ("cuda", "xpu"):
        return False
    mlp_width = max(
        config.intermediate_size,
        getattr(config, "gen_moe_intermediate_size", 0) * getattr(config, "gen_num_experts_per_tok", 1),
    )
    tokens = 2 * input_embeds.shape[0] * input_embeds.shape[1]
    workspace = tokens * input_embeds.element_size() * (12 * config.hidden_size + 4 * mlp_width)
    if activation_budget is not None and workspace > activation_budget:
        return False
    accel = getattr(torch, input_embeds.device.type)
    try:
        free = accel.mem_get_info(input_embeds.device)[0]
        # Do not count allocator caches as free: cudaMallocAsync pools can be
        # stream-specific, and pending prefetch may still need those blocks.
        return free >= workspace + 2 * 1024**3
    except (RuntimeError, AttributeError):
        return False
