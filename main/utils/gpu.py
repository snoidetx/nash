import os
import torch


def use_gpus_(ids=[0]):
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, ids))


def connect_to_(index=0):
    if torch.cuda.is_available():
        device = torch.device("cuda", index=index)
    else:
        device = torch.device("cpu")

    torch.cuda.set_device(device)
    print(f"Connected to {device}")
    return device
    