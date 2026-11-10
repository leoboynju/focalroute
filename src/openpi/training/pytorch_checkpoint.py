"""CPU validation of this project's optimizer checkpoints."""

import numpy as np
import torch


def load_cpu_optimizer(path):
    # The cosine LR schedule saves numpy.float64 after warmup.
    allowed = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
