"""Helpers used by training config files.

Defined in a stable importable module (not inside the config .py itself) so
that any function references captured into ``gpc.config`` remain picklable
when checkpoints serialize the full config via ``torch.save``.
"""
import os


def env_bool(name, default=False):
    """Parse a 'true'/'false' env var to bool. Empty/missing -> ``default``."""
    raw = os.environ.get(name, '').strip().lower()
    if raw == '':
        return default
    return raw in ('1', 'true', 'yes', 'y', 'on')
