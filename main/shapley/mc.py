import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from main.utils.saveload import save, load


PATH = os.path.dirname(__file__)


def _eval_one_perm(perm, utility_function, X_train, y_train, X_val, y_val):
    n_data = len(y_train)
    subsets, utilities = [], []
    for j in range(1, n_data+1):
        subset_indices = perm[:j]
        subsets.append(subset_indices)
        utilities.append(utility_function(X_train[subset_indices], y_train[subset_indices], X_val, y_val))
    
    return subsets, utilities


def monte_carlo_shapley(utility_function, X_train, y_train, X_val, y_val, n_perms=100, 
                        save_samples=None,
                        load_samples=None,
                        show_progress=True, 
                        seed=42,
                        n_jobs=16,
                        backend="loky",
                        max_nbytes="64M",
                        batch_size="auto"):
    n_data = len(y_train)
    subsets = []
    utilities = []
    samples = [subsets, utilities]
    base_utility = utility_function(X_train[[]], y_train[[]], X_val, y_val)
    if not load_samples:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(n_data) for _ in range(n_perms)]
        result = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size, max_nbytes=max_nbytes)(
                     delayed(_eval_one_perm)(p, utility_function, X_train, y_train, X_val, y_val)
                         for p in tqdm(perms, desc="Permutations", disable=not show_progress))
        samples[0] = [s for subset_list, _ in result for s in subset_list]
        samples[1] = [u for _, util_list in result for u in util_list]

        if save_samples:
            save(samples, os.path.join(PATH, "..", "..", "saved_data", "samples", f"{save_samples}.pkl"))

    if load_samples:
        samples = load(os.path.join(PATH, "..", "..", "saved_data", "samples", f"{load_samples}.pkl"))
        if len(samples[0]) != n_perms * n_data:
            raise AttributeError("No. of permutations do not match no. of loaded samples")

    subsets = samples[0]
    utilities = samples[1]
    m = 1 if not hasattr(utilities[0], "__len__") else len(utilities[0])
    shape = (n_data, m) if m > 1 else (n_data,)
    shapleys = np.zeros(shape, dtype=float)
    for i in range(len(subsets)):
        if i % n_data == 0:
            shapleys[subsets[i][-1]] += utilities[i] - base_utility
        else:
            shapleys[subsets[i][-1]] += utilities[i] - utilities[i-1]

    ps_shapleys = shapleys / n_perms
    shapleys = np.mean(ps_shapleys, axis=1)

    return shapleys, ps_shapleys
