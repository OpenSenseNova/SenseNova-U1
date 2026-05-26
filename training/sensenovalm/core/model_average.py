# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
EMA/SWA averaged model implementation that is safe with distributed ProcessGroup.

The stock `torch.optim.swa_utils.AveragedModel` deepcopies the whole model at init,
but `torch.distributed.ProcessGroup` is not pickle/deepcopy-able. In this codebase,
several modules/communicators may keep ProcessGroup references, so deepcopy(model)
can crash.

This implementation avoids deepcopy(model). Instead, it keeps a shadow copy of
tracked tensors (parameters and optionally buffers).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

MultiAvgFn = Callable[[List[Tensor], List[Tensor], Union[Tensor, int]], None]
AvgFn = Callable[[Tensor, Tensor, Union[Tensor, int]], Tensor]


def _iter_named_params_and_buffers(model: Module, use_buffers: bool) -> Iterable[Tuple[str, Tensor]]:
    for n, p in model.named_parameters(recurse=True):
        yield n, p
    if use_buffers:
        for n, b in model.named_buffers(recurse=True):
            yield n, b


class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    Exponential Moving Average is a variation of `Polyak averaging`_,
    but using exponential weights instead of equal weights across iterations.

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA/EMA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        multi_avg_fn (function, optional): the averaging function used to update
            parameters inplace; the function must take in the current values of the
            :class:`AveragedModel` parameters as a list, the current values of :attr:`model`
            parameters as a list, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with the `avg_fn` or `multi_avg_fn` parameters.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights (SWA).

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_model = torch.optim.swa_utils.AveragedModel(model,
        >>>             torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)

    .. note::
        When using SWA/EMA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` and `multi_avg_fn` are not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    .. _Polyak averaging:
        https://paperswithcode.com/method/polyak-averaging
    """

    n_averaged: Tensor

    def __init__(
        self,
        model: Module,
        device: Optional[Union[int, torch.device]] = None,
        avg_fn: Optional[AvgFn] = None,
        multi_avg_fn: Optional[MultiAvgFn] = None,
        use_buffers: bool = False,
    ):  # noqa: D107
        super().__init__()
        assert avg_fn is None or multi_avg_fn is None, "Only one of avg_fn and multi_avg_fn should be provided"

        self.use_buffers = use_buffers
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))

        # Store tracked tensor names and shadow values as numbered buffers: avg_0, avg_1, ...
        self._names: List[str] = []
        self._kinds: List[str] = []  # "param" or "buffer"

        buf_names = set(dict(model.named_buffers(recurse=True)).keys())
        for name, t in _iter_named_params_and_buffers(model, use_buffers=use_buffers):
            self._names.append(name)
            self._kinds.append("buffer" if name in buf_names else "param")
            shadow = t.detach().clone()
            if device is not None:
                shadow = shadow.to(device)
            self.register_buffer(f"avg_{len(self._names) - 1}", shadow)

    def get_extra_state(self) -> Dict[str, object]:
        return {"names": self._names, "kinds": self._kinds, "use_buffers": self.use_buffers}

    def set_extra_state(self, state: Dict[str, object]) -> None:
        self._names = list(state.get("names", []))
        self._kinds = list(state.get("kinds", []))
        self.use_buffers = bool(state.get("use_buffers", False))

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "This AveragedModel does not own a deepcopy-ed module. "
            "Use `copy_to(model)` or `with ema.apply_to(model): ...` for evaluation."
        )

    def update_parameters(self, model: Module):
        avg_tensors = [getattr(self, f"avg_{i}") for i in range(len(self._names))]
        params = dict(model.named_parameters(recurse=True))
        bufs = dict(model.named_buffers(recurse=True)) if self.use_buffers else {}
        model_tensors: List[Tensor] = []
        for name in self._names:
            if name in params:
                model_tensors.append(params[name])
            elif name in bufs:
                model_tensors.append(bufs[name])
            else:
                raise KeyError(f"Tracked tensor '{name}' not found in model {model.state_dict().keys()} ")

        if int(self.n_averaged.item()) == 0:
            for a, m in zip(avg_tensors, model_tensors):
                a.copy_(m.detach().to(a.device))
            self.n_averaged += 1
            return

        n = self.n_averaged
        if self.multi_avg_fn is not None:
            self.multi_avg_fn(
                avg_tensors,
                [m.detach().to(avg_tensors[0].device) for m in model_tensors],
                n,
            )
        elif self.avg_fn is not None:
            for a, m in zip(avg_tensors, model_tensors):
                a.copy_(self.avg_fn(a, m.detach().to(a.device), n))
        else:
            for a, m in zip(avg_tensors, model_tensors):
                m_ = m.detach().to(a.device)
                a.add_(m_ - a, alpha=1.0 / float(n.item() + 1))

        self.n_averaged += 1

    @torch.no_grad()
    def get_averaged_model_state_dict(self, model: Module) -> Dict[str, Tensor]:
        """
        Build a deepcopy-free state_dict for saving:
        - Start from model.state_dict() (so keys are complete)
        - Override tracked params/buffers with averaged tensors
        """
        sd: Dict[str, Tensor] = model.state_dict()
        avg_tensors = [getattr(self, f"avg_{i}") for i in range(len(self._names))]
        for name, a in zip(self._names, avg_tensors):
            if name not in sd:
                # model structure changed; keep strict to avoid silently writing wrong ckpt
                raise KeyError(f"Tracked tensor '{name}' not found in model.state_dict()")
            ref = sd[name]
            sd[name] = a.detach().to(ref.device, dtype=ref.dtype).view_as(ref)
        return sd

    @torch.no_grad()
    def load_from_model_state_dict(self, model_state_dict: Dict[str, Tensor]) -> None:
        """
        Initialize EMA/SWA shadow averaged model weights from a model-like state_dict
        (e.g. loaded from `<ckpt>/averaged_model/`).
        This avoids touching `model` and keeps ProcessGroup out of deepcopy.
        """
        avg_tensors = [getattr(self, f"avg_{i}") for i in range(len(self._names))]
        for name, a in zip(self._names, avg_tensors):
            if name not in model_state_dict:
                raise KeyError(f"averaged_model state dict missing key '{name}'")
            v = model_state_dict[name]
            a.copy_(v.detach().to(device=a.device, dtype=a.dtype).view_as(a))
        # Mark as initialized so subsequent update_parameters uses averaging path, not copy.

    @torch.no_grad()
    def load_from_checkpoint_folder(
        self,
        model: Module,
        folder: str,
        *,
        state_dict_loader: Callable[[str, Module], Dict[str, Tensor]],
    ) -> None:
        """
        Resume EMA/SWA shadow averaged model weights from a checkpoint folder which stores model weight shards
        (e.g. `<ckpt>/averaged_model/`).

        Abstraction:
        - `state_dict_loader(folder, model)` handles sharding/TP/PP/MoE specifics and returns a model-like state_dict
          for the current rank.
        - This method only consumes the state_dict and fills EMA/SWA shadow buffers.
        """
        sd = state_dict_loader(folder, model)
        info_path = os.path.join(folder, "averaged_model_info.json")
        with open(info_path, "r") as f:
            info = json.load(f)
        self.n_averaged.fill_(info["n_averaged"])
        self.load_from_model_state_dict(sd)

    def save_to_averaged_model_info(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        info_path = os.path.join(folder, "averaged_model_info.json")
        with open(info_path, "w") as f:
            json.dump({"n_averaged": self.n_averaged.item()}, f)

    @torch.no_grad()
    def copy_to(self, model: Module):
        avg_tensors = [getattr(self, f"avg_{i}") for i in range(len(self._names))]
        params = dict(model.named_parameters(recurse=True))
        bufs = dict(model.named_buffers(recurse=True)) if self.use_buffers else {}
        model_tensors: List[Tensor] = []
        for name in self._names:
            if name in params:
                model_tensors.append(params[name])
            elif name in bufs:
                model_tensors.append(bufs[name])
            else:
                raise KeyError(f"Tracked tensor '{name}' not found in model {model.state_dict().keys()}")
        for a, m in zip(avg_tensors, model_tensors):
            m.copy_(a.detach().to(m.device))

    @contextmanager
    def apply_to(self, model: Module):
        params = dict(model.named_parameters(recurse=True))
        bufs = dict(model.named_buffers(recurse=True)) if self.use_buffers else {}
        model_tensors: List[Tensor] = []
        for name in self._names:
            if name in params:
                model_tensors.append(params[name])
            elif name in bufs:
                model_tensors.append(bufs[name])
            else:
                raise KeyError(f"Tracked tensor '{name}' not found in model (model structure changed?)")
        backup = [t.detach().clone() for t in model_tensors]
        try:
            self.copy_to(model)
            yield model
        finally:
            with torch.no_grad():
                for t, b in zip(model_tensors, backup):
                    t.copy_(b.to(t.device))