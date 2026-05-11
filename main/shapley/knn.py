import numpy as np
from tqdm import tqdm

from main.shapley.helpers.helper_knn import rank_neighbor, knn_shapley_JW_single, get_tuned_K


def knn_shapley(X_train, y_train, X_val, y_val, K=5, dis_metric='cosine'):
    n_train = len(y_train)
    n_val = len(y_val)
    shapleys = np.zeros(n_train)
    ps_shapleys = np.zeros((n_train, n_val))
    for i in tqdm(range(n_val)):
        X_val_single, y_val_single = X_val[i], y_val[i]
        shapley = knn_shapley_JW_single(X_train, y_train, X_val_single, y_val_single, K=K, dis_metric=dis_metric)
        shapleys += shapley
        ps_shapleys[:, i] = shapley
  
    return shapleys, ps_shapleys
