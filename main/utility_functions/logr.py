import numpy as np
from sklearn.linear_model import LogisticRegression


def get_logr_utility(X_train, y_train, X_val, y_val, per_sample=True, n_classes=None):
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    if not n_classes:
        n_classes = len(np.unique(y_val))

    if len(y_train) == 0:
        if not per_sample:
            return 1/n_classes
        else:
            return np.full(len(y_val), 1/n_classes)

    try:
        model.fit(X_train, y_train)
    except:
        if not per_sample:
            return 1/n_classes
        else:
            return np.full(len(y_val), 1/n_classes)

    if not per_sample:
        acc = model.score(X_val, y_val)
        return acc
    else:
        y = model.predict(X_val)
        return (y == y_val).astype(int)


def get_logr_utility_conditional(X_train, y_train, X_val, y_val, X_selected, y_selected, weight=None):
    model = LogisticRegression(max_iter=5000, solver='liblinear')
    X_train = np.concatenate((X_train, X_selected), axis=0)
    y_train = np.concatenate((y_train, y_selected), axis=0)
    
    if len(y_train) == 0:
        return 0.5

    try:
        model.fit(X_train, y_train, sample_weight=weight)
    except:
        return 0.5

    acc = model.score(X_val, y_val)
    return acc


import numpy as np
from sklearn.linear_model import LogisticRegression


def get_logr_loss_utility(X_train, y_train, X_val, y_val, per_sample=True, n_classes=None):
    model = LogisticRegression(max_iter=1000, solver='liblinear')

    if n_classes is None:
        n_classes = len(np.unique(y_val))

    # fallback utility when training fails / empty train set
    # uniform prediction => log loss = -log(1 / n_classes) = log(n_classes)
    # so negated loss = -log(n_classes)
    fallback = -np.log(n_classes)

    if len(y_train) == 0:
        if not per_sample:
            return fallback
        else:
            return np.full(len(y_val), fallback)

    try:
        model.fit(X_train, y_train)
    except Exception:
        if not per_sample:
            return fallback
        else:
            return np.full(len(y_val), fallback)

    classes = model.classes_
    probs = model.predict_proba(X_val)  # shape: (n_val, n_seen_classes)

    # align probabilities to all classes in y_val if needed
    if n_classes is None:
        all_classes = np.unique(y_val)
    else:
        all_classes = np.arange(n_classes)

    full_probs = np.zeros((len(y_val), len(all_classes)))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    for j, c in enumerate(classes):
        if c in class_to_idx:
            full_probs[:, class_to_idx[c]] = probs[:, j]

    # probability assigned to the true class
    y_idx = np.array([class_to_idx[c] for c in y_val])
    true_probs = full_probs[np.arange(len(y_val)), y_idx]

    # avoid log(0)
    eps = 1e-12
    neg_losses = np.log(np.clip(true_probs, eps, 1.0))   # = - cross-entropy per sample

    if not per_sample:
        return neg_losses.mean()
    else:
        return neg_losses