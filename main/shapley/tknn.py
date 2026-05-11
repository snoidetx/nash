import numpy as np
from tqdm import tqdm

from main.shapley.helpers.helper_knn import tnn_shapley_single, get_tuned_tau


def threshold_knn_shapley(X_train, y_train, X_val, y_val, tau=-0.5):
    n_train = len(y_train)
    n_val = len(y_val)
    shapleys = np.zeros(n_train)
    ps_shapleys = np.zeros((n_train, n_val))
    for i in tqdm(range(n_val)):
        X_val_single, y_val_single = X_val[i], y_val[i]
        shapley = tnn_shapley_single(X_train, y_train, X_val_single, y_val_single, tau=tau)
        shapleys += shapley
        ps_shapleys[:, i] = shapley

    return shapleys, ps_shapleys

