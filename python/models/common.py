import time
import torch


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def read_class_label(labels_file):
    with open(labels_file, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]
