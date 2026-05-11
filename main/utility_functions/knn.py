import numpy as np
from tqdm import tqdm


def rank_neighbors(x_val, X_train):
    distance = -np.dot(X_train, x_val) / np.linalg.norm(X_train, axis=1)
    return np.argsort(distance)


def get_knn_utility(X_train, y_train, X_val, y_val, K=5, per_sample=False):
    if not per_sample:
        result = 0
    else:
        result = np.zeros(len(y_val))

    for i in range(len(y_val)):
        rank = rank_neighbors(X_val[i], X_train)
        acc_single = 0
        n_neighbors = min(K, len(y_train))
        for j in range(n_neighbors):
            acc_single += int(y_val[i] == y_train[rank[j]]) / n_neighbors

        if not per_sample:
            result += acc_single
        else:
            result[i] = acc_single

    return result / len(y_val) if not per_sample else result
