# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Originally adapted from DeepSpeed (Microsoft, Apache-2.0); routing logic
# influenced by the Switch Transformer paper (Fedus et al., 2021).
"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.modules.mlp import new_feed_forward
from sensenovalm.utils.logger import get_logger

from .base_layer import BaseMoELayer
from .monitor import MoEMonitor
from .utils import (
    all_to_all,
    gather_along_first_dim_expert_parallel,
    gather_from_parallel_region_to_moe,
    moe_gather,
    moe_scatter,
    reduce_scatter_to_parallel_region_from_moe,
)

sensenovalm_accelerator = get_accelerator()

try:
    # To enable gemm permute optimizations on GPU:
    #   python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.3
    import grouped_gemm

    GEMM_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    # Fail silently so we don't spam logs unnecessarily if user isn't using gemm
    GEMM_INSTALLED = False
    pass

# global llm logger
logger = get_logger(__file__)

sensenovalm_accelerator = get_accelerator()

uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device), high=torch.tensor(1.0 + epsilon, device=device)
        ).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def custom_argsort(x, stable=True):
    if sensenovalm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        sorted_indices = torch.sort(x.to(torch.float), stable=stable)[1]
        return sorted_indices
    else:
        return torch.argsort(x, stable=stable)


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk: int = 1,
        noisy_gate_policy: Optional[str] = None,
        scoring_func: str = "sigmoid",
        jitter_eps: float = 1e-2,
        gate_monitor: Optional[MoEMonitor] = None,
        dtype: Optional[torch.dtype] = torch.float,
    ) -> None:
        super().__init__()

        # Deepspeed's mechisms, alway use fp32
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = topk
        self.jitter_eps = jitter_eps

        self.noisy_gate_policy = noisy_gate_policy
        self.gate_monitor = gate_monitor

        self.scoring_func = scoring_func
        self.reset_parameters()

    def reset_parameters(self) -> None:
        from torch.nn import init

        init.kaiming_uniform_(self.wg.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        # input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(inputs, epsilon=self.jitter_eps, device=inputs.device)
        logits = self.wg(inputs)
        if self.gate_monitor is not None and not gpc.is_forward_done:
            self.gate_monitor.pre_gate_profiling(logits)
        if self.training and gpc.config.loss.moe_z_loss_coef > 0:
            z_loss = torch.logsumexp(logits, dim=1).square().mean()
        else:
            z_loss = None
        if self.scoring_func == "sigmoid":
            gates = logits.sigmoid()
        elif self.scoring_func == "softmax":
            gates = F.softmax(logits, dim=1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        return gates, logits, z_loss


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class QwenDroplessMoELayer(BaseMoELayer):
    """MoELayer module which implements MixtureOfExperts as described in Gshard_."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        activation_type: str = "swiglu",
        drop_and_pad: bool = False,
        drop_policy="probs",
        capacity_factor: float = None,
        noisy_gate_policy: str = None,
        moe_grouped_mlp: bool = True,
        enable_fused_permute: bool = True,
        token_dispatch_policy: str = "alltoall",
        deterministic_mode: bool = False,
        moe_jitter_eps: float = 1e-2,
        normalize_expert_weights: bool = True,
        routed_scaling_factor: float = 1.0,
        scoring_func: str = "softmax",
        moe_monitor_cfg: Optional[Dict] = None,
        **_ignored,
    ) -> None:
        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )
        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"

        if moe_monitor_cfg is not None:
            self.expert_monitor = MoEMonitor(
                ep_size=ep_size,
                num_experts=num_experts,
                topk=top_k,
                moe_monitor_cfg=moe_monitor_cfg,
            )
        else:
            self.expert_monitor = None
        backend = "bmm" if drop_and_pad else "gmm"
        if moe_grouped_mlp:
            experts = new_feed_forward(
                in_features,
                hidden_features,
                out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
                is_expert=True,
                use_grouped_mlp=True,
                num_groups=num_experts // ep_size,
                backend=backend,
            )
        else:
            experts = torch.nn.ModuleList(
                [
                    new_feed_forward(
                        in_features,
                        hidden_features,
                        out_features,
                        bias=False,
                        device=device,
                        dtype=dtype,
                        mlp_layer_fusion=mlp_layer_fusion,
                        multiple_of=multiple_of,
                        activation_type=activation_type,
                        is_expert=True,
                    )
                    for _ in range(num_experts // ep_size)
                ]
            )
        super().__init__(
            TopKGate(
                in_features,
                num_experts,
                top_k,
                noisy_gate_policy,
                scoring_func,
                moe_jitter_eps,
                gate_monitor=self.expert_monitor,
                dtype=dtype
            ),
            experts,
            ep_group,
            ep_size,
            num_experts // ep_size,
        )

        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        local_expert_indices_offset = gpc.get_local_rank(ParallelMode.EXPERT) * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        self.topk = top_k
        self.moe_grouped_mlp = moe_grouped_mlp
        self.enable_fused_permute = (
            GEMM_INSTALLED and enable_fused_permute and not drop_and_pad and capacity_factor is None
        )
        self.deterministic_mode = deterministic_mode
        self.normalize_expert_weights = normalize_expert_weights
        self.routed_scaling_factor = routed_scaling_factor
        self.drop_and_pad = drop_and_pad
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        if self.drop_and_pad:
            assert self.capacity_factor is not None
        self.capacity = None

        self.token_dispatch_policy = token_dispatch_policy
        if self.token_dispatch_policy == "alltoall":
            self.token_permutation_func = self.token_permutation_by_alltoall
            self.token_unpermutation_func = self.token_unpermutation_by_alltoall
            self.enable_fused_permute = (
                GEMM_INSTALLED and enable_fused_permute and not drop_and_pad and capacity_factor is None
            )
            self.input_splits = None
            self.output_splits = None
            self.num_global_tokens_per_local_expert_cpu = None
            # We need to keep track of the token num if we drop tokens without padding them.
            self.num_out_tokens = None
            self.hidden_shape = None
            # A cuda stream synchronization is needed due to no blocking sync between host and device
            self.device_sync_point = "no_sync"
            input_chunk_idxs = torch.arange(self.num_experts)  # * self.tp_size
            # [num_local_experts, ep_size]. Sort the input chunks by local experts.
            self.sort_input_by_local_experts = input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel().tolist()
            # [ep_size, num_local_experts]. Restore the output chunks by local experts.
            self.restore_output_by_local_experts = (
                input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel().tolist()
            )

        elif self.token_dispatch_policy == "allgather":
            self.token_permutation_func = self.token_permutation_by_all_gather
            self.token_unpermutation_func = self.token_unpermutation_by_all_gather
            # self.local_probs: probs of global token assignment to local experts.
            self.local_probs = None
            # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
            # each element is True if it's between the local_expert_indices. Only useful when cross
            # device token permutation is enabled and **AllGahter** is performed.
            self.global_local_map = None
            assert (
                not drop_and_pad and capacity_factor is None
            ), "allgather dispatcher do not support drop_and_pad, please use all2all"
        else:
            assert False, "unsupported token dispatch policy"

    def forward(self, *inputs: Tensor) -> Tensor:
        self.hidden_shape = inputs[0].shape

        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        gates, gate_logits, z_loss = self.gate(reshaped_inputs)
        scores_for_choice = gates
        expert_weights, indices, tokens_per_expert_before_capacity = self.topk_softmax_with_capacity(
            gates, scores_for_choice
        )

        (dispatched_input, tokens_per_expert) = self.token_permutation_func(
            reshaped_inputs, expert_weights, indices, tokens_per_expert_before_capacity
        )
        if self.moe_grouped_mlp:
            expert_output = self.experts(dispatched_input, batch_sizes=tokens_per_expert)
        else:
            expert_output = self.experts(dispatched_input, split_size_or_sections=tokens_per_expert, split_dim=0)
        output = self.token_unpermutation_func(expert_output, expert_weights)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        return output, gate_logits, z_loss

    def topk_softmax_with_capacity(self, gates, scores_for_choice):
        _, indices = torch.topk(scores_for_choice, self.topk, dim=1)
        expert_weights = gates.gather(1, indices)
        if self.normalize_expert_weights:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        # we compute num_local_tokens_per_expert here. If no drop and padding, num_local_tokens_per_expert should be
        # the final value, otherwise we recompute it in self.process(.)
        # histc(.) can be faster the bincount(.), but will cause non-deterministic behavior
        if self.deterministic_mode:
            num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        else:
            num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)

        expert_weights = expert_weights * self.routed_scaling_factor
        # without capacity
        if self.capacity_factor is None:
            return expert_weights, indices, num_local_tokens_per_expert

        # with capacity
        expert_capacity = get_capacity(
            num_tokens=gates.shape[0] * self.topk,
            num_experts=gates.shape[1],
            capacity_factor=self.capacity_factor,
        )
        # TopK selection, Maskout unused experts
        # TODO Here we use expert_weights to drop tokens, or maybe we should use biased weight?
        topk_masked_gates = torch.zeros_like(gates).scatter(1, indices, expert_weights)
        topk_mask = torch.zeros_like(gates).scatter(1, indices, 1)
        if self.drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
        elif self.drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        if self.drop_and_pad:
            final_expert_weights, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, indices)
            final_expert_weights = expert_weights * torch.logical_not(exceed_mask)
            final_indices = indices.clone().masked_fill_(exceed_mask, torch.iinfo(torch.long).max)

        tokens_per_expert_before_capacity = topk_mask.sum(dim=0)

        return final_expert_weights, final_indices, tokens_per_expert_before_capacity

    def preprocess(self, indices, expert_weight, tokens_per_expert_before_capacity) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes
        the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        # NOTE: bincount seem slower than histc
        # num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        if self.capacity_factor is not None:
            if self.deterministic_mode:
                num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
            else:
                num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)
        else:
            num_local_tokens_per_expert = tokens_per_expert_before_capacity


        if self.drop_and_pad:
            self.capacity = expert_weight.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            self.num_global_tokens_per_local_expert_cpu = torch.full(
                (self.num_experts,), self.capacity, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.capacity_factor is not None:
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(torch.device("cpu"), non_blocking=True)
            self.device_sync_point = "before_permutation_1"
        elif self.ep_size > 1 or self.num_local_experts > 1:
            # wait for input_splits and output_splits, or num_global_tokens_per_local_expert_cpu sync
            self.device_sync_point = "before_ep_alltoall"
        else:
            # wait for tokens_per_expert sync
            self.device_sync_point = "before_premute_finish"

        if self.ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )

            num_global_tokens_per_expert = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert).reshape(
                self.ep_size, self.num_experts
            )
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices]

            # NOTE:another impl for num_global_tokens_per_local_expert calc, seems spent same time
            # self.num_global_tokens_per_local_expert = torch.empty_like(num_local_tokens_per_expert)
            # torch.distributed.all_to_all_single(
            #     self.num_global_tokens_per_local_expert, num_local_tokens_per_expert,
            #     group=gpc.get_group(ParallelMode.EXPERT), async_op=False
            # )
            # self.num_global_tokens_per_local_expert = self.num_global_tokens_per_local_expert.
            # reshape(self.ep_size, -1)

            self.output_splits = (
                num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
            )
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(self.num_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.expert_monitor is not None and not gpc.is_forward_done:
            self.expert_monitor.expert_profiling(
                tokens_per_expert_before_capacity=(
                    tokens_per_expert_before_capacity if self.capacity_factor is not None else None
                ),
                num_tokens_per_expert=(
                    num_global_tokens_per_expert.sum(axis=0) if self.ep_size > 1 else num_tokens_per_local_expert
                ),
            )

        if self.moe_grouped_mlp:
            num_tokens_per_local_expert = num_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)

        if self.num_local_experts > 1 and self.ep_size > 1:
            self.num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            ).to(torch.device("cpu"), non_blocking=True)

        return num_tokens_per_local_expert

    def permute_with_padded_tokens(self, tokens, indices):
        """Permute the tokens based on the indices, only used in padding mode.
        The input indices shape is [num_expert, capacity], it indicates which tokens were selected
        by each expert separately.
        Args:
            tokens (torch.Tensor): The input token tensor.
            indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the
                                    selected tokens for each expert.

        Returns:
            torch.Tensor: The permuted tensor.
            torch.Tensor: The sorted_indices corresponding permuted tensor.
        """
        permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

        return permuted_tokens, indices

    def unpermute_with_padded_tokens(
        self,
        permuted_tokens: torch.Tensor,
        indices: torch.Tensor,
        expert_weights: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unpermutes a padded permuted tokens based on sorted indices and merges the tokens
        with their corresponding expert_weights.

        This function takes a tensor of permuted tokens and reorders them according to the provided indices.
        It also combines the tokens with their associated expert_weights.

        Parameters:
            permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
            indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for
                                    each expert.
            expert_weights (torch.Tensor): A tensor with the same shape as indices, containing weights
                                           corresponding to each token.
            restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

        Returns:
            torch.Tensor: A tensor of unpermuted tokens, merged with their expert_weights.

        """
        # Ensure permuted_tokens is 2D
        assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

        # Reshape and expand expert_weights and indices to match permuted_tokens
        expert_weights = expert_weights.view(-1).unsqueeze(-1)

        # Combine tokens with their expert_weights
        combined_output = expert_weights * permuted_tokens

        # Prepare a tensor of zeros with the desired output shape
        empty_tokens = torch.zeros(
            restore_shape,
            dtype=combined_output.dtype,
            device=combined_output.device,
        )

        # Scatter the combined tokens back to their original positions
        flatten_indices = indices.view(-1)
        unpermuted_tokens = empty_tokens.index_put_((flatten_indices,), combined_output, accumulate=True)

        return unpermuted_tokens

    def permute(self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
        """Permute the tokens based on the indices. Token with the same index will be grouped together.
        The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
        Args:
            tokens (torch.Tensor): The input token tensor.
            indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or
                        [num_tokens, topk].
            num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should
                        equal the number of tokens not dropped.  By default, set to None, meaning no tokens are dropped.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.

        Returns:
            torch.Tensor: The permuted tensor.
            torch.Tensor: The sorted_indices corresponding permuted tensor.
        """
        if padded_mode:
            return self.permute_with_padded_tokens(tokens, indices)

        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = custom_argsort(flatten_indices, stable=True)

        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices

    def unpermute(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        expert_weights: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
    ):
        """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens
           with their corresponding expert_weights.

        Args:
            permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
            sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
            expert_weights (torch.Tensor, optional): The tensor of expert_weights corresponding to the permuted tokens.
                        If provided, the unpermuted tokens will be merged with their respective expert_weights.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.
            restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode.
                        Defaults to None.

        Returns:
            torch.Tensor: The unpermuted tokens, optionally merged with expert_weights.
        """
        if padded_mode:
            return self.unpermute_with_padded_tokens(
                permuted_tokens, sorted_indices, expert_weights, restore_shape=restore_shape
            )

        assert sorted_indices.numel() == permuted_tokens.size(0)
        if expert_weights is not None:
            # Unpermute and merge the tokens with their expert_weights
            num_unpermuted_tokens = expert_weights.numel()
            topk = expert_weights.size(1)
        else:
            # Unpermute the tokens without merge
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_put_((sorted_indices,), permuted_tokens, accumulate=False)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
        if expert_weights is not None:
            unpermuted_tokens = unpermuted_tokens * expert_weights.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens

    def sort_chunks_by_idxs(self, inputs: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
        """Split and sort the input tensor based on the split_sizes and sorted indices."""
        inputs = torch.split(inputs, split_sizes.tolist(), dim=0)
        output = torch.cat([inputs[i] for i in sorted_idxs], dim=0)
        return output

    def token_permutation_by_alltoall(
        self,
        reshaped_inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert_before_capacity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            expert_weights (torch.Tensor): expert_weights of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        assert expert_weights.dim() == 2, "Expected 2D tensor for expert weights"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(indices, expert_weights, tokens_per_expert_before_capacity)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = reshaped_inputs.shape
        if self.device_sync_point == "before_permutation_1":
            sensenovalm_accelerator.current_stream().synchronize()
        if self.enable_fused_permute:
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = grouped_gemm.ops.permute(
                reshaped_inputs, indices.to(torch.int32), self.num_out_tokens
            )
        else:
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = self.permute(
                reshaped_inputs,
                indices,
                num_out_tokens=self.num_out_tokens,
                padded_mode=self.drop_and_pad,
            )

        if self.expert_monitor is not None and not gpc.is_forward_done:
            self.expert_monitor.expert_profiling_wait()

        # Perform expert parallel AlltoAll communication
        if self.device_sync_point == "before_ep_alltoall":
            sensenovalm_accelerator.current_stream().synchronize()
        global_input_tokens, _ = all_to_all(
            permutated_local_input_tokens, self.output_splits, self.input_splits, gpc.get_group(ParallelMode.EXPERT)
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1 and self.ep_size > 1:
            global_input_tokens = self.sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
            )
        if self.device_sync_point == "before_premute_finish":
            sensenovalm_accelerator.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation_by_alltoall(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1 and self.ep_size > 1:
            hidden_states = self.sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]

        permutated_local_input_tokens, _ = all_to_all(
            hidden_states, self.input_splits, self.output_splits, gpc.get_group(ParallelMode.EXPERT)
        )

        # Unpermutation 1: AlltoAll output to output
        if self.enable_fused_permute:
            output = grouped_gemm.ops.unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                expert_weights.to(torch.float32),
            )
        else:
            output = self.unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                expert_weights=expert_weights,
                padded_mode=self.drop_and_pad,
                restore_shape=self.hiddden_shape_before_permute,
            )

        return output

    def token_permutation_by_all_gather(
        self,
        reshaped_inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert_before_capacity: torch.Tensor,  # pylint: disable=W0613
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            reshaped_inputs: 3D tensor [S/TP, B, H]. Input tokens.
            expert_weights: 2D tensor [S/TP*B, topk]. Each row of expert_weights contains
            the probility distribution across `topk` experts for one local token.
            For 'aux_loss' load balancing, the sum of the values in each row is 1,
            thus for `top1` gating, it degenerates into a full 1 tensor.
            indices: 2D tensor [num_local_tokens, topk], where
            `num_local_tokens=S/TP*B`. Token assignment to global experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """

        # num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)
        # Permute the tokens across the expert parallel devices.
        if self.ep_size > 1:
            # local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, topk] -> [num_global_tokens, topk], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                global_indices = gather_from_parallel_region_to_moe(indices)
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_mask)

            # local_probs calculation
            # expert_weights: [S/TP*B, topk] -> global_probs: [S*B*EP, topk]
            global_probs = gather_from_parallel_region_to_moe(expert_weights)
            self.local_probs = global_probs.masked_select(global_local_mask)
            self.local_probs = self.local_probs.view(-1, 1)
            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            global_hidden_states = gather_from_parallel_region_to_moe(reshaped_inputs)  # , use_global_buffer=True
            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, reshaped_inputs.shape[-1])
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
        else:
            if self.topk > 1:
                global_local_mask = torch.ones_like(indices).bool()
                local_indices = indices.masked_select(global_local_mask)
                self.local_probs = expert_weights.masked_select(global_local_mask)
                self.local_probs = self.local_probs.view(-1, 1)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(-1, reshaped_inputs.shape[-1])
                local_hidden_states = torch.gather(reshaped_inputs, 0, self.global_local_map)
            else:
                local_indices = indices
                self.local_probs = expert_weights.view(-1, 1)
                local_hidden_states = reshaped_inputs
                self.global_local_map = None

        with torch.no_grad():
            tokens_per_expert = torch.histc(local_indices.view(-1), bins=self.num_experts, min=0, max=self.num_experts)
            if self.num_local_experts < self.num_experts:
                tokens_per_expert = tokens_per_expert[self.local_expert_indices[0] : self.local_expert_indices[-1] + 1]
            if self.moe_grouped_mlp:
                tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather

        # if self.enable_fused_permute:
        #     permuted_local_hidden_states, self.reversed_local_input_permutation_mapping = grouped_gemm.ops.permute(
        #         reshaped_inputs, indices.to(torch.int32)
        #     )
        # else:
        permuted_local_hidden_states, self.reversed_local_input_permutation_mapping = self.permute(
            local_hidden_states, local_indices
        )

        return permuted_local_hidden_states, tokens_per_expert

    def token_unpermutation_by_all_gather(
        self, hidden_states: torch.Tensor, expert_weight: torch.Tensor = None
    ):  # pylint: disable=W0613
        """
        Reverse process of `dispatch()` which permutes the output of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor [num_permuted_tokens_for_local_experts, H],
            output of local experts.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [S/TP, B, H]
        """
        # Stage1: unpermute the tokens locally.
        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.

        # if self.enable_fused_permute:
        #     unpermuted_local_hidden = grouped_gemm.ops.unpermute(
        #         hidden_states,
        #         self.reversed_local_input_permutation_mapping)
        # else:
        unpermuted_local_hidden = self.unpermute(hidden_states, self.reversed_local_input_permutation_mapping)
        unpermuted_local_hidden = unpermuted_local_hidden * self.local_probs

        output_total = unpermuted_local_hidden

        # Unpermute the tokens across expert parallel devices.
        if self.ep_size > 1:
            assert self.global_local_map is not None, "global_local_map is necessary for `AllGather`."
            # hidden_shape: [S/TP, B, H], gloal_num_tokens = S/TP*B*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * self.ep_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = moe_scatter.apply(
                unpermuted_local_hidden, self.global_local_map, global_hidden_shape
            )
            output_total = reduce_scatter_to_parallel_region_from_moe(unpermuted_global_hidden)
        else:
            if self.topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(0, self.global_local_map, unpermuted_local_hidden)

        return output_total
