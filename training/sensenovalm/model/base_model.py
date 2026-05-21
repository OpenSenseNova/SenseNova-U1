# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
from abc import ABCMeta, abstractmethod

from torch import nn

from sensenovalm.model.utils import load_src_states, merge_pp_src_states


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for all models.
    """

    @staticmethod
    @abstractmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load_sharded_states(src):
        states = merge_pp_src_states(load_src_states(src))
        num_shards = len(states)
        return states, num_shards
