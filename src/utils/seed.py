import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets random seed for reproducibility.
    Ensures same results across runs where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # For deterministic behavior (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Global seed set to: {seed}")
