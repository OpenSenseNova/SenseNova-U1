from torch import nn


def set_parallel_attribute(module, parallel_attribute):
    if isinstance(module, nn.Parameter):
        setattr(module, parallel_attribute, True)
        return
    for param in module.parameters():
        setattr(param, parallel_attribute, True)
