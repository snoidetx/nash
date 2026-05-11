import numpy as np
from sklearn.linear_model import Ridge


def get_rr_utility(X_train, y_train, X_val, y_val, per_sample=True, alpha=1.0):
    model = Ridge(alpha=alpha)

    if len(y_train) == 0:
        if not per_sample:
            return -np.mean(y_val.astype(float) ** 2)
        else:
            return -(y_val.astype(float) ** 2)

    try:
        model.fit(X_train, y_train)
    except:
        if not per_sample:
            return -np.mean(y_val.astype(float) ** 2)
        else:
            return -(y_val.astype(float) ** 2)

    y_pred = model.predict(X_val)
    se = (y_pred - y_val) ** 2

    if not per_sample:
        return -np.mean(se)
    else:
        return -se
