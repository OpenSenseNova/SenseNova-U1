# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from typing import Callable, Dict

import torch

from sensenovalm.accelerator import get_accelerator
from sensenovalm.utils.logger import get_logger

sensenovalm_accelerator = get_accelerator()
logger = get_logger(__file__)

uniform_map: Dict[torch.device, Callable] = {}


class BaseMonitor:
    """
    Monitor base class for monitoring MoE experts.
    """

    def __init__(self):
        """
        Initializes the BaseMonitor with a common configuration for monitoring.

        """
        self.handles = []

    def clear_handles(self):
        """Clear all asynchronous handles."""
        for handle in self.handles:
            handle.wait()

        self.handles = []
