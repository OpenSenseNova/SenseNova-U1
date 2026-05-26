# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
communication for tensor/sequence parallel.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple

import torch
from torch import distributed as dist

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.core.parallel.comm.utils import (
    DUMMY_HANDLE_CONST,
    AsyncCommHandle,
    _gather,
    _split,
    all_gather_raw,
    all_reduce_raw,
    gather_forward_split_backward,
    reduce_forward,
    reduce_scatter_raw,
    split_forward_gather_backward,
)
from sensenovalm.model.modules.embedding import Embedding1D
from sensenovalm.model.moe.moe import MoE

# input gather dim
_GATHER_DIM = 1  # shape: [batch, seqlen, dim] or [1, packlen, dim]
_REDUCE_DIM = 1  # shape: [batch, seqlen, dim] or [1, packlen, dim]


class LinearRole(Enum):
    COLUMN = "column"
    ROW = "row"


# not really useful, only for code hint.
class TPCommunicator(ABC):
    """
    Common communicator interafce for tensor/sequence parallel.
    """

    @abstractmethod
    def save_total_input(self) -> bool:
        """
        Should linear save total input after all gather as activation in sequence parallel.
        """
        pass

    @abstractmethod
    def communication_mode(self) -> str:
        """
        communication mode of communictor
        """
        pass

    @abstractmethod
    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for input when forward/backward.
        """
        pass

    @abstractmethod
    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for grad_output when backward.
        """
        pass

    @abstractmethod
    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for grad_input when backward.
        """
        pass

    @abstractmethod
    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        communication for output when forward.
        """
        pass


class TensorParallelCommunicator(TPCommunicator):
    """
    tensor parallel communicator for linear
    """

    def __init__(self, process_group: dist.ProcessGroup, role: LinearRole) -> None:
        assert role in (LinearRole.COLUMN, LinearRole.ROW), f"Unknown linear role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = False

    def save_total_input(self) -> bool:
        return self._save_total_input

    def communication_mode(self) -> str:
        return "tp"

    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for input.
        """
        return _input, DUMMY_HANDLE_CONST

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        tensor parallel should do nothing for grad_output.
        """
        return grad_output, DUMMY_HANDLE_CONST

    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return all_reduce_raw(grad_input, process_group=self._process_group, async_op=async_op)

    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all reduce output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return all_reduce_raw(output, process_group=self._process_group, async_op=async_op)


class SequenceParallelCommunicator(TPCommunicator):
    """
    sequence parallel communicator for linear
    """

    def __init__(
        self, process_group: dist.ProcessGroup, role: LinearRole, save_total_input_as_activation: bool = False
    ) -> None:
        assert role in (LinearRole.COLUMN, LinearRole.ROW), f"Unknown linear role: {role}"

        self._process_group = process_group
        self._role = role

        self._save_total_input = save_total_input_as_activation

    def save_total_input(self) -> bool:
        return self._save_total_input

    def communication_mode(self) -> str:
        return "sp"

    def input_hook(
        self, _input: torch.Tensor, async_op: bool = False, is_forward: bool = True
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather input only for column parallel linear when forward/backward.
        """
        # 1. world_size <= 1
        # 2. row parallel linear should not allgather input.
        # 3. column parallel linear should not allgather input if save_total_input_as_activation and backward is True.
        if (
            dist.get_world_size(self._process_group) <= 1
            or self._role == LinearRole.ROW
            or (is_forward is False and self._save_total_input)
        ):
            return _input, DUMMY_HANDLE_CONST

        return all_gather_raw(_input, process_group=self._process_group, async_op=async_op, gather_dim=_GATHER_DIM)

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather grad_output only for row parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return grad_output, DUMMY_HANDLE_CONST

        return all_gather_raw(grad_output, process_group=self._process_group, async_op=async_op, gather_dim=_GATHER_DIM)

    def grad_input_hook(self, grad_input: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter grad_input only for column parallel linear when backward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.ROW:
            return grad_input, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(
            grad_input, process_group=self._process_group, async_op=async_op, reduce_dim=_REDUCE_DIM
        )

    def output_hook(self, output: torch.Tensor, async_op: bool = False) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        reduce scatter output only for row parallel linear when forward.
        """
        if dist.get_world_size(self._process_group) <= 1 or self._role == LinearRole.COLUMN:
            return output, DUMMY_HANDLE_CONST

        return reduce_scatter_raw(output, process_group=self._process_group, async_op=async_op, reduce_dim=_REDUCE_DIM)


class HeadTensorParallelCommunicator(TensorParallelCommunicator):
    """
    tensor parallel communicator for head linear
    """

    def __init__(self, parallel_mode: ParallelMode, retain_out_sharded: bool = True) -> None:
        super().__init__(process_group=gpc.get_group(parallel_mode), role=LinearRole.COLUMN)

        self._parallel_mode = parallel_mode
        self._retain_out_sharded = retain_out_sharded

    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        split grad_output if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return grad_output, DUMMY_HANDLE_CONST

        return _split(grad_output, parallel_mode=self._parallel_mode, dim=-1), DUMMY_HANDLE_CONST

    def output_hook(
        self, output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather output for head layer if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return output, DUMMY_HANDLE_CONST

        return _gather(output, parallel_mode=self._parallel_mode, dim=-1), DUMMY_HANDLE_CONST


class HeadSequenceParallelCommunicator(SequenceParallelCommunicator):
    """
    sequence parallel communicator for head linear
    """

    def __init__(
        self, parallel_mode: ParallelMode, retain_out_sharded: bool = True, save_total_input_as_activation: bool = False
    ) -> None:
        super().__init__(
            process_group=gpc.get_group(parallel_mode),
            role=LinearRole.COLUMN,
            save_total_input_as_activation=save_total_input_as_activation,
        )

        self._parallel_mode = parallel_mode
        self._retain_out_sharded = retain_out_sharded

    # rewrite grad_output communication hook
    def grad_output_hook(
        self, grad_output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        split grad_output if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return grad_output, DUMMY_HANDLE_CONST

        return _split(grad_output, parallel_mode=self._parallel_mode, dim=-1), DUMMY_HANDLE_CONST

    # rewrite ouput communication hook
    def output_hook(
        self, output: torch.Tensor, async_op: bool = False  # pylint: disable=W0613
    ) -> Tuple[torch.Tensor, AsyncCommHandle]:
        """
        all gather output for head layer if retain_out_sharded is False.
        """
        if self._retain_out_sharded or dist.get_world_size(self._process_group) <= 1:
            return output, DUMMY_HANDLE_CONST

        return _gather(output, parallel_mode=self._parallel_mode, dim=-1), DUMMY_HANDLE_CONST


class MoESequenceParallelCommunicator:
    """
    sequence parallel communicator for moe layer
    parallel_mode: sequence parallel mode
    reverse: gather input when enter moe and split output when leave if set False,
             otherwise do the opposite operation
    """

    def __init__(self, parallel_mode: ParallelMode, reverse: bool = False) -> None:
        self._parallel_mode = parallel_mode
        self.reverse = reverse

    def register_module_hook(self, module: MoE) -> None:
        assert isinstance(module, MoE), "MoE sequence parallel communicator is only support moe module"

        module.register_forward_pre_hook(self.input_hook, with_kwargs=True)
        module.register_forward_hook(self.output_hook)

    def input_hook(self, module: MoE, args, kwargs) -> torch.Tensor:  # pylint: disable=W0613
        """
        allgather input before forward and split grad_input after backward.
        """
        _input = args[0] if len(args) > 0 else kwargs.pop("hidden_states")
        args = args[1:] if len(args) > 0 else args
        if self.reverse:
            _input = split_forward_gather_backward(_input, self._parallel_mode, dim=_REDUCE_DIM)
        else:
            _input = gather_forward_split_backward(_input, self._parallel_mode, dim=_GATHER_DIM)

        return (_input, *args), kwargs

    def output_hook(self, module: MoE, args: Any, output: Tuple[Any]) -> Tuple[Any]:  # pylint: disable=W0613
        """
        split output after forward and allgather grad_output before backward.
        """
        _output, *_others = output
        if self.reverse:
            _output = gather_forward_split_backward(_output, self._parallel_mode, dim=_GATHER_DIM)
        else:
            _output = split_forward_gather_backward(_output, self._parallel_mode, dim=_REDUCE_DIM)

        return (_output, *_others)


class EmbeddingTensorParallelCommunicator:
    """
    tensor parallel communicator for embbeding layer
    """

    def __init__(self, parallel_mode: ParallelMode) -> None:
        self._parallel_mode = parallel_mode

    def register_module_hook(self, module: Embedding1D) -> None:
        assert isinstance(module, Embedding1D), "Embbeding tensor parallel communicator is only support Embedding1D"

        module.register_forward_hook(self.output_hook)

    def output_hook(self, module: Embedding1D, args: Any, output: Tuple[Any]) -> Tuple[Any]:  # pylint: disable=W0613
        """
        split output after forward and allgather grad_output before backward.
        """
        _emb_dim = 2  # [bsz, seqlen, emb_dim]

        if module.vocab_parallel:
            output = reduce_forward(output, self._parallel_mode)
        else:
            output = gather_forward_split_backward(output, self._parallel_mode, dim=_emb_dim)

        return output


class EmbeddingSequenceParallelCommunicator:
    """
    sequence parallel communictor for embbeding layer
    """

    def __init__(self, parallel_mode: ParallelMode) -> None:
        self._parallel_mode = parallel_mode

    def register_module_hook(self, module: Embedding1D) -> None:
        assert isinstance(module, Embedding1D), "Embbeding sequence parallel communicator is only support Embedding1D"

        module.register_forward_hook(self.output_hook)

    def output_hook(self, module: Embedding1D, args: Any, output: Tuple[Any]) -> Tuple[Any]:  # pylint: disable=W0613
        """
        split output after forward and allgather grad_output before backward.
        """
        _emb_dim, _seq_dim = 2, 1  # [bsz, seqlen, emb_dim]

        # tp:
        if module.vocab_parallel:
            output = reduce_forward(output, self._parallel_mode)
        else:
            output = gather_forward_split_backward(output, self._parallel_mode, dim=_emb_dim)
        # sp:
        output = split_forward_gather_backward(output, self._parallel_mode, dim=_seq_dim)

        return output
