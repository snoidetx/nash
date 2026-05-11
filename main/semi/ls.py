import os
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy.special import comb

from main.utils.saveload import save, load


PATH = os.path.dirname(__file__)


def _eval_one_hund(hund, utility_function, X_train, y_train, X_val, y_val):
    n_data = len(y_train)
    subsets, utilities = [], []
    base_utility = utility_function(X_train[[]], y_train[[]], X_val, y_val)
    for subset_indices in hund:
        subsets.append(subset_indices)
        subset_indices_new = subset_indices[subset_indices != n_data]
        utilities.append(utility_function(X_train[subset_indices_new], y_train[subset_indices_new], X_val, y_val) - base_utility)
    
    return subsets, utilities


def ls_semivalue(utility_function, X_train, y_train, X_val, y_val,
                 weights, # sum to 1
                 n_hunds=100, 
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
    weights = (weights / comb(n_data - 1, np.arange(n_data))) * comb(n_data + 1, np.arange(1, n_data + 1))
    normalized_weights = weights / np.sum(weights)
    if not load_samples:
        rng = np.random.default_rng(seed)
        hunds = [[rng.choice(n_data+1, size=rng.choice(np.arange(1, n_data+1), p=normalized_weights), replace=False) for _ in range(100)] for _ in range(n_hunds)]
        result = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size, max_nbytes=max_nbytes)(
                     delayed(_eval_one_hund)(p, utility_function, X_train, y_train, X_val, y_val)
                         for p in tqdm(hunds, desc="Hundreds", disable=not show_progress))
        samples[0] = [s for subset_list, _ in result for s in subset_list]
        samples[1] = [u for _, util_list in result for u in util_list]

        if save_samples:
            save(samples, os.path.join(PATH, "..", "..", "saved_data", "samples", f"{save_samples}.pkl"))

    if load_samples:
        samples = load(os.path.join(PATH, "..", "..", "saved_data", "samples", f"{load_samples}.pkl"))
        if len(samples[0]) != n_hunds * 100:
            raise AttributeError("No. of hundreds do not match no. of loaded samples")

    subsets = samples[0]
    utilities = np.array(samples[1], dtype=float)
    m = 1 if not hasattr(utilities[0], "__len__") else len(utilities[0])
    shape = (n_data+1, m) if m > 1 else (n_data,)
    shapleys = np.zeros(shape, dtype=float)
    counts = np.zeros(n_data + 1, dtype=float)
    for i in tqdm(range(len(subsets))):
        counts[subsets[i]] += 1
        mask = np.zeros(n_data + 1)
        mask[subsets[i]] = 1
        shapleys += mask[:, None] * utilities[i][None, :]
 
    ps_shapleys = shapleys / counts[:, None]
    w = np.sum(np.arange(1, n_data+1)/ (n_data+1) * weights)
    ps_shapleys = w * ((ps_shapleys - ps_shapleys[-1])[:-1, :])
    shapleys = np.mean(ps_shapleys, axis=1)

    return shapleys, ps_shapleys
