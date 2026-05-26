# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
communication for zero parallel
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Set, Union

import torch
from torch import distributed as dist
from torch import nn

from sensenovalm.core.naive_amp import NaiveAMPModel
from sensenovalm.core.parallel.comm.isp import ISPCommunicator
from sensenovalm.model.mtp import MTP
from sensenovavl.model.sensenovavl_moe_chat.modeling_neo_vit import InternVisionEncoder


class ParamAsyncBcastHandler:
    """
    Model Partition Handler for overlap broadcast with forward
    """

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        max_zero_size: int,
        queue_length: int = 8,
    ) -> None:
        self._block_to_param: Dict[nn.Module, List[nn.Parameter]] = OrderedDict()
        self._param_to_rank: Dict[nn.Parameter, int] = {}
        self._block_to_rank: Dict[nn.Module, Set[int]] = {}
        self._bcast_handles: Dict[int, List[dist.Work]] = {}
        self._bcast_closures: Dict[int, List[Callable]] = {}
        self._bcast_queue: List[dist.Work] = []
        self._bcast_rank_queue_closed: Dict[bool] = None

        self._bcast_queue_length = queue_length
        self._max_zero_size = max_zero_size

        total_param_num = sum(p.numel() for p in model.parameters())
        avg_param_num = total_param_num * 1.0 // max_zero_size

        # initialize an empty list for _bcast_handles of each rank
        self._bcast_handles = {rank: [] for rank in range(max_zero_size)}
        self._bcast_closures = {rank: [] for rank in range(max_zero_size)}
        self._bcast_rank_queue_closed = {rank: True for rank in range(max_zero_size)}
        self._bcast_rank_queue_closed[None] = True  # gloabl key: None

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        # record the parameters to transformer/embeding/head/norm block
        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model

            for _, multimodal in _chunk.named_children():
                if isinstance(multimodal, torch.nn.Sequential):
                    self._block_to_param[multimodal] = list(multimodal.parameters())
                    continue

                for _, children in multimodal.named_children():
                    # should be the transformer block definaton in modeling_xxx.py
                    if isinstance(children, nn.ModuleList):
                        # record the block that a parameter belongs to
                        for _, block in enumerate(children):
                            self._block_to_param[block] = list(block.parameters())
                    elif isinstance(children, InternVisionEncoder):
                        for _, block in enumerate(children.layers):
                            self._block_to_param[block] = list(block.parameters())
                    elif isinstance(children, MTP):
                        for _, block in enumerate(children.mtp_layers):
                            self._block_to_param[block] = list(block.parameters())
                    else:
                        # record the block that a parameter belongs to
                        # self._block_to_param[name] = list(children.parameters())
                        self._block_to_param[children] = list(children.parameters())

        alloc_num = 0
        rank_to_go = 0

        # process the parameters in block_to_param sequencially,
        # allocate each parameter to a local rank of ParallelMode.ZERO1,
        # NOTE that we do NOT consider following scenarios:
        # 1) whether a parameter is trainable;
        # 2) paramters maybe in different optimizer group
        for block, params in self._block_to_param.items():
            # allocate a model block to a local rank of ParallelMode.ZERO1
            for p in params:
                alloc_num = alloc_num + p.numel()
                # in this case, allocate the param to next rank if possible
                if alloc_num > avg_param_num * 1.01 and rank_to_go < max_zero_size - 1:
                    rank_to_go = rank_to_go + 1
                    alloc_num = 0
                # allocate a parameter to a local rank of ParallelMode.ZERO1
                self._param_to_rank[p] = rank_to_go

    def partition_param_list(self, param_list: list, partition_size: int):
        if partition_size != self._max_zero_size:
            if len(param_list) == 0:
                _max_rank_allocated = 0
            else:
                _max_rank_allocated = max([self._param_to_rank[p] for p in param_list]) + 1
            if _max_rank_allocated > partition_size:
                for param in param_list:
                    self._param_to_rank[param] = self._param_to_rank[param] * partition_size // _max_rank_allocated

        no_params_ranks = []
        params_per_rank = [[] for _ in range(partition_size)]
        numel_per_rank = [0 for _ in range(partition_size)]
        params_per_rank_id_dict = [[] for _ in range(partition_size)]

        for i, param in enumerate(param_list):
            if param.requires_grad is False:
                continue

            global_id = str(i)
            for j in range(len(param.size())):
                global_id = "_".join([global_id, str(param.size()[j])])

            _rank = self._param_to_rank[param]
            params_per_rank[_rank].append(param)
            numel_per_rank[_rank] += param.numel()
            params_per_rank_id_dict[_rank].append(global_id)

        for rank, params in enumerate(params_per_rank):
            if len(params) == 0:
                no_params_ranks.append(rank)

        return params_per_rank, set(no_params_ranks), numel_per_rank, params_per_rank_id_dict

    def _resync_block_to_rank(self) -> None:
        for block, params in self._block_to_param.items():
            self._block_to_rank[block] = set()
            for param in params:
                self._block_to_rank[block].add(self._param_to_rank[param])

    def register_sync_parameters_hook(self, isp_communicator: ISPCommunicator = None) -> None:
        # assert isp_communicator is None, "isp is nyi"
        # resync block to rank
        self._resync_block_to_rank()

        def _pre_forward_hook(model: nn.Module, *args, **kwargs):  # pylint: disable=W0613
            bcast_handles = []
            # gather all required broadcast hanles into a list
            for rank in self._block_to_rank[model]:
                self.submit_bcast_async(submit_rank=rank)
                bcast_handles.extend(self._bcast_handles[rank])
                # need to clear _bcast_handles since they would be processed later
                self._bcast_handles[rank] = []
            # wait all required broadcast handles to be completed
            for handle in bcast_handles:
                handle.wait()

            self.submit_bcast_async()

        # register_forward_pre_hook for transformer/embeding/norm/xxx block
        for block, _ in self._block_to_param.items():
            # NOTE: Although the layernorm layer does not have explicit processing,
            # both ISPCommunicator and ParamAsyncBcastHandler handle transformer blocks as granularity,
            # so everything is fine.
            block.register_forward_pre_hook(_pre_forward_hook)

        if isp_communicator:
            isp_communicator.register_prerequisite_for_forward_prefetch_hooks(_pre_forward_hook)

    def add_bcast_closure(self, rank, closure) -> None:
        if rank not in self._bcast_closures:
            self._bcast_closures[rank] = []

        self._bcast_rank_queue_closed[rank] = False
        self._bcast_rank_queue_closed[None] = False
        self._bcast_closures[rank].append(closure)

    def submit_bcast_async(
        self,
        manual_submit_count: int = None,
        submit_all: bool = False,
        submit_rank: int = None,
    ) -> None:
        def _get_next_closure(_rank: int = None) -> Callable:
            _old_rank = _rank
            if _rank is None:
                _rank = min(self._bcast_closures.keys(), default=None)

            if _rank is None or _rank not in self._bcast_closures:
                self._bcast_rank_queue_closed[_rank] = True
                return None, None

            try:
                return _rank, self._bcast_closures[_rank].pop(0)
            except IndexError:
                self._bcast_closures.pop(_rank)
                self._bcast_rank_queue_closed[_rank] = True
                return _get_next_closure(_old_rank)

        if self._bcast_rank_queue_closed[submit_rank]:
            return

        # remove completed handles
        self._bcast_queue = list(filter(lambda x: not x.is_completed(), self._bcast_queue))

        if submit_all:
            submit_count = len(self._bcast_queue) + sum((len(x) for x in self._bcast_closures.values()))
        elif submit_rank is not None:
            submit_count = len(self._bcast_queue) + len(self._bcast_closures[submit_rank])
        else:
            submit_count = self._bcast_queue_length if manual_submit_count is None else manual_submit_count

        if len(self._bcast_queue) >= submit_count:
            return

        for _ in range(submit_count - len(self._bcast_queue)):
            rank, closure = _get_next_closure(submit_rank)
            if closure is None:
                break

            handle = closure()

            self._bcast_queue.append(handle)
            self._bcast_handles[rank].append(handle)
