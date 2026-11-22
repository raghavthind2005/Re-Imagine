"""Deterministic seeding for python, numpy and (when installed) torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every random number generator the project uses.

    Torch is optional so that pure-logic code and tests run without it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """Return an independent numpy generator, for code that should not touch global state."""
    return np.random.default_rng(seed)
