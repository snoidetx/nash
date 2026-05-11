"""
This code is partially adapted from https://github.com/Jiachen-T-Wang/data-banzhaf/tree/main.
"""

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from tqdm import tqdm

from main.utils.saveload import save, load


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATASET_NAMES = ['wind', 'pol', 'phoneme', 'cpu', '2dplanes', 'apsfailure', 'vehicle', 'creditcard']
CIFAR10_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_dataset(name, n_train=200, n_val=2000, flip_ratio=0, use_embedding=False, device='cpu'):
    if name not in DATASET_NAMES:
        raise ValueError(f"Invalid dataset name: {name}")

    np.random.seed(999)
    if name == 'wind':
        return load_wind(n_train, n_val, flip_ratio)
    elif name == 'pol':
        return load_pol(n_train, n_val, flip_ratio)
    elif name == 'phoneme':
        return load_phoneme(n_train, n_val, flip_ratio)
    elif name == 'cpu':
        return load_cpu(n_train, n_val, flip_ratio)
    elif name == '2dplanes':
        return load_2dplanes(n_train, n_val, flip_ratio)
    elif name == 'apsfailure':
        return load_apsfailure(n_train, n_val, flip_ratio)
    elif name == 'vehicle':
        return load_vehicle(n_train, n_val, flip_ratio)
    elif name == 'creditcard':
        return load_creditcard(n_train, n_val, flip_ratio)


def load_2dplanes(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, '2dplanes_727.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y']
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets = make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_apsfailure(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'APSFailure_41138.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y']
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets = make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_vehicle(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'vehicle_sensIT_357.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y']
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets = make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_creditcard(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'CreditCardFraudDetection_42397.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y']
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets = make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val
    

def load_wind(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'wind_847.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y']
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets = make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_pol(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'pol_722.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y'] # data[:,1:], data[:,0]
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets=make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_phoneme(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'phoneme_1489.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y'] # data[:,1:], data[:,0]
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets=make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def load_cpu(n_train, n_val, flip_ratio):
    dataset_raw = load(os.path.join(DATASET_PATH, 'cpu_761.pkl'))
    data, targets = dataset_raw['X_num'], dataset_raw['y'] # data[:,1:], data[:,0]
    targets = (targets == 1) + 0.0
    targets = targets.astype(np.int32)
    data, targets=make_balanced_dataset(data, targets)
    idxs=np.random.permutation(len(data))
    data, targets=data[idxs], targets[idxs]
    X_train, y_train, X_val, y_val = split_train_val(data, targets, n_train, n_val)
    y_train = flip_label(y_train, flip_ratio)

    return X_train, y_train, X_val, y_val


def make_balanced_dataset(data, targets):
    """Returns a balanced dataset."""

    p = np.mean(targets)
    if p < 0.5:
        minor_class = 1
    else:
        minor_class = 0
    
    index_minor_class = np.where(targets == minor_class)[0]
    n_minor_class = len(index_minor_class)
    n_major_class = len(targets) - n_minor_class
    new_minor = np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)
    data=np.concatenate([data, data[new_minor]])
    targets=np.concatenate([targets, targets[new_minor]])

    return data, targets


def split_train_val(X, y, n_train, n_val):
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_mean, X_std= np.mean(X_train, 0), np.std(X_train, 0)
    normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
    X_train, X_val = normalizer_fn(X_train), normalizer_fn(X_val)

    return X_train, y_train, X_val, y_val


def flip_label(y, flip_ratio):
    np.random.seed(999)
    n_flip = int(len(y) * flip_ratio)
    n_class = len(np.unique(y))

    if n_class == 2:
        y[:n_flip] = 1 - y[:n_flip]
    else:
        y[:n_flip] = np.array([np.random.choice(np.setdiff1d(np.arange(n_class), [y[i]])) for i in range(n_flip)])

    return y
