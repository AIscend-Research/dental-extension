"""Reproducibility helpers. Call set_seed() at the top of every experiment."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy, and (if present) torch. Keep experiments comparable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic cudnn trades a little speed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
