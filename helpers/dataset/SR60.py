import os.path as p
import glob

import torch

from typing import Union

from torch.utils.data import Dataset


class SR60(Dataset):
    def __init__(
        self,
        path: str,
        limit: Union[int, None] = None,
    ):
        self.path = p.join(path, "SR60")

        if limit:
            print("Warning: a dataset limit has been set")

        self.limit = limit

    def __len__(self):
        l = len(glob.glob(p.join(self.path, "060", "*.pt")))

        if self.limit is not None:
            return min(l, self.limit)
        return l

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p1, p2, p3, p1_eval = (
            torch.load(p.join(self.path, "060", f"{idx}.pt")),
            torch.load(p.join(self.path, "120", f"{idx}.pt")),
            torch.load(p.join(self.path, "360", f"{idx}.pt")),
            torch.load(p.join(self.path, "060", "eval", f"{idx}.pt")),
        )

        return (p1, p2, p3), p1_eval
