# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0); accelerator interface
# inspired by DeepSpeed (Microsoft, Apache-2.0).
"""
Universal accelerator interface implementation, inspired by DeepSpeed.
"""

import enum
import os


class AcceleratorType(enum.Enum):
    GPU = 1
    NPU = 2
    CPU = 3
    DIPU = 4
    DITORCH = 5
    OTHER = 6


sensenovalm_accelerator = None


class Accelerator:
    """
    Abstract base class for accelerator
    """

    def __init__(self) -> None:
        pass

    def get_backend_name(self):
        """
        Return the name of the accelerator.
        """
        raise NotImplementedError

    def get_accelerator_backend(self):
        """
        Return the name of the backend.
        """
        raise NotImplementedError

    # Device APIs
    def device_name(self, device_index=None):
        """
        Return the name of the device.
        """
        raise NotImplementedError

    def set_device(self, device_index):
        """
        Bind the current process to a device.
        """
        raise NotImplementedError

    def get_device_id(self):
        """
        Return the current device index.
        """
        raise NotImplementedError

    def current_device_name(self):
        """
        Return the name of the current device.
        """
        raise NotImplementedError

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        raise NotImplementedError

    def synchronize(self, device_index=None):
        """
        Synchronize the current process.
        """
        raise NotImplementedError


def get_accelerator():
    # This (open-source) build only supports the CUDA backend.
    global sensenovalm_accelerator
    if sensenovalm_accelerator is not None:
        return sensenovalm_accelerator

    accelerator_name = os.environ.get("SENSENOVALM_ACCELERATOR", "cuda")
    if accelerator_name != "cuda":
        raise ValueError(
            f"Only the 'cuda' accelerator is supported, but SENSENOVALM_ACCELERATOR='{accelerator_name}'."
        )

    from .cuda_accelerator import CUDA_Accelerator

    sensenovalm_accelerator = CUDA_Accelerator()
    return sensenovalm_accelerator
