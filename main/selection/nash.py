import math
from tqdm import tqdm

import numpy as np
import torch

from main.utils.random import set_random_seed


@torch.no_grad()
def _find_best_idx_parallel(current_sum: torch.Tensor,
                            X: torch.Tensor,
                            remaining_mask: torch.Tensor,
                            concave_fn,
                            base_obj: float):
    idxs = torch.nonzero(remaining_mask, as_tuple=False).view(-1)
    candidates = X[idxs]                        # (n_rem, p)
    sums = candidates + current_sum.unsqueeze(0)  # (n_rem, p)

    obj_vals = concave_fn(sums).sum(dim=1)             # (n_rem,)
    gains = obj_vals - base_obj                # (n_rem,)

    loc = gains.argmax()
    best_idx = idxs[loc].item()
    best_gain = gains[loc].item()
    return best_idx, best_gain


@torch.no_grad()
def nash_selection(
    X: torch.Tensor,
    k: int,
    concave_fn=None,
    device: torch.device = None,
    seed: int = None,
) -> np.ndarray:
    if concave_fn is None:
        raise ValueError("Please provide `concave_fn`: a callable(tensor)->tensor (concave & nondecreasing).")

    if seed is not None:
        set_random_seed(seed)

    if device is None:
        device = X.device
    X = X.to(device)

    n, p = X.shape
    if k > n:
        raise ValueError("k cannot exceed the number of vectors.")

    B = torch.zeros(p, device=device, dtype=X.dtype)
    current_sum = B.clone()
    remaining = torch.ones(n, dtype=torch.bool, device=device)

    selected = []

    for _ in tqdm(range(k)):
        base_obj = concave_fn(current_sum).sum().item()

        best_idx, _ = _find_best_idx_parallel(
            current_sum=current_sum,
            X=X,
            remaining_mask=remaining,
            concave_fn=concave_fn,
            base_obj=base_obj
        )

        selected.append(best_idx)
        current_sum = current_sum + X[best_idx]
        remaining[best_idx] = False

    return np.array(selected)
