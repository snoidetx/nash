import numpy as np

from main.shapley.helpers.helper_knn import get_tnn_acc


def get_threshold_knn_utility(X_train, y_train, X_val, y_val, tau=-0.5):
    return get_tnn_acc(X_train, y_train, X_val, y_val, tau=tau)
