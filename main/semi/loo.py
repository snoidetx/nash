import os
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy.special import comb

from main.utils.saveload import save, load


def loo_value(utility_function, X_train, y_train, X_val, y_val,
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
    base_utility = utility_function(X_train[[]], y_train[[]], X_val, y_val)
    full_utility = utility_function(X_train, y_train, X_val, y_val)
    loos = []
    for i in tqdm(range(n_data)):
        subset = [j for j in range(n_data) if j != i]
        utility_minus_i = utility_function(X_train[subset], y_train[subset], X_val, y_val)
        loo_i = full_utility - utility_minus_i
        loos.append(loo_i)

    ps_loos = np.array(loos)
    loos = np.mean(ps_loos, axis=1)

    return loos, ps_loos
