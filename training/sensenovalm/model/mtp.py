# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import torch
from torch import nn

from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.model.modules.linear import new_linear
from sensenovalm.model.modules.norm import new_layer_norm
from sensenovalm.solver.activation_checkpoint import activation_checkpoint


class MTPDecoder(nn.Module):
    """
    Multi-Token Prediction decoder block.

    Wraps an existing decoder (provided by the caller) with norm + concat +
    projection so a single MTP head can predict an additional token offset
    from the main backbone output.

    Args:
        hidden_size (int): The size of hidden state.
        decoder (nn.Module): The decoder block instance to apply after projection.
        layer_norm_epsilon (float): RMSNorm epsilon.
        checkpoint (bool): If True, use activation checkpointing during training.
    """

    def __init__(
        self,
        hidden_size,
        decoder: nn.Module,
        layer_norm_epsilon: float = 1e-5,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.norm_before_output = new_layer_norm("rmsnorm", hidden_size, eps=layer_norm_epsilon)
        self.norm_after_embedding = new_layer_norm("rmsnorm", hidden_size, eps=layer_norm_epsilon)
        self.proj = new_linear("out_proj", 2 * hidden_size, hidden_size, bias=False)
        self.layer = decoder
        self.checkpoint = checkpoint

    def forward(self, hidden_states_before_output, hidden_states_after_embedding):
        if self.checkpoint and self.training:
            return activation_checkpoint(
                self._forward, False, hidden_states_before_output, hidden_states_after_embedding
            )
        else:
            return self._forward(hidden_states_before_output, hidden_states_after_embedding)

    def _forward(self, hidden_states_before_output, hidden_states_after_embedding):
        hidden_states_left = self.norm_before_output(hidden_states_before_output)
        hidden_states_right = self.norm_after_embedding(hidden_states_after_embedding)
        hidden_states = torch.cat([hidden_states_left, hidden_states_right], dim=-1)
        proj_output = self.proj(hidden_states)
        hidden_states, moe_outputs = self.layer(proj_output)
        return hidden_states, moe_outputs


class MTP(nn.Module):
    """
    Multi-Token Prediction head: stacks one or more MTPDecoder blocks to
    predict tokens offset by 1, 2, ... from the main next-token target.

    Each decoder shares the backbone's embedding, final norm, and output layer.

    Args:
        hidden_size (int): The size of hidden state.
        decoders (list[tuple[nn.Module, bool]]): Per-MTP-layer (decoder, checkpoint) pairs.
        layer_norm_epsilon (float): RMSNorm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        decoders,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.mtp_layers = nn.ModuleList(
            [MTPDecoder(hidden_size, decoders[i][0], layer_norm_epsilon, decoders[i][1]) for i in range(len(decoders))]
        )
        assert hasattr(gpc.config, "pad_token_id"), "pad_token_id should be set in gpc.config"
        self.pad_token_id = gpc.config.pad_token_id

    def forward(self, hidden_states, input_ids, shared_embedding_layer, shared_norm, shared_output_layer):
        shared_embeding = shared_embedding_layer
        shared_output = shared_output_layer
        mtp_outputs = []
        moe_outputs = []

        for index, mtp_block in enumerate(self.mtp_layers):
            input_ids_trunc = torch.cat(
                [
                    input_ids[:, index + 1 :],
                    torch.full(
                        (input_ids.size(0), index + 1),
                        self.pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            hidden_states_after_embedding = shared_embeding(input_ids_trunc)
            hidden_states, cur_moe_output = mtp_block(hidden_states, hidden_states_after_embedding)
            hidden_states = shared_norm(hidden_states)
            mtp_output = shared_output(hidden_states)
            mtp_outputs.append(mtp_output)
            moe_outputs.append(cur_moe_output)
        return mtp_outputs, moe_outputs
