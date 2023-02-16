import os
import os.path as p
import glob
import json

import torch

import torchvision.transforms as T

from typing import Union, List

from torch.utils.data import Dataset

from helpers.sentinel import Resolution


class SR20(Dataset):
    def __init__(
        self,
        path: str,
        limit: Union[int, None] = None,
    ):
        self.path = p.join(path, "SR20")

        if limit:
            print("Warning: a dataset limit has been set")

        self.limit = limit

    def __len__(self):
        l = len(glob.glob(p.join(self.path, "20", "*.pt")))

        if self.limit is not None:
            return min(l, self.limit)
        return l

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p1, p2, p1_eval = (
            torch.load(p.join(self.path, "20", f"{idx}.pt")),
            torch.load(p.join(self.path, "40", f"{idx}.pt")),
            torch.load(p.join(self.path, "20", "eval", f"{idx}.pt")),
        )

        return (p1, p2), p1_eval
