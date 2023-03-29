import os.path as p
import glob
import json

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset

from typing import Union, Dict, List

from helpers.log import error


def _t(t):
    return torch.tensor(t)


def _t_to_float():
    return T.Lambda(lambda x: x.float())


def _t_to_int16():
    return T.Lambda(lambda x: torch.round(x).to(torch.int16))


def normalize_t(norm: Dict[str, Dict[str, List[float]]], resolution: str):
    return T.Compose(
        [
            _t_to_float(),
            T.Normalize(mean=norm["mean"][resolution], std=norm["std"][resolution]),
        ]
    )


def denormalize_t(norm: Dict[str, Dict[str, List[float]]], resolution: str):
    mean, std = _t(norm["mean"][resolution]), _t(norm["std"][resolution])
    return T.Compose(
        [
            T.Normalize(mean=-(mean / std), std=1 / std),
            _t_to_int16(),
        ]
    )


class SR20(Dataset):
    def __init__(
        self,
        path: str,
        limit: Union[int, None] = None,
        dtype: Union[torch.dtype, None] = torch.float16,
        normalization=True,
    ):
        self.path = p.join(path, "SR20")

        if limit:
            print(f"Warning: a dataset limit of {limit} patches has been set")

        self.limit = limit

        self.norm = None
        self.norm_t = None

        if normalization:
            self.norm = norm_SR20
            self.norm_t = [
                normalize_t(self.norm, "10"),
                normalize_t(self.norm, "20"),
                normalize_t(self.norm, "20"),
            ]
        else:
            if dtype != torch.int16:
                dtype = torch.int16
                print(
                    "Warning: if normalization is disabled, dtype is automatically set to int16"
                )

        self.dtype = dtype

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

        if self.norm_t is not None:
            p1, p2, p1_eval = (
                self.norm_t[0](p1),
                self.norm_t[1](p2),
                self.norm_t[2](p1_eval),
            )

        if self.dtype is not None:
            if self.dtype != p1.dtype:
                p1 = p1.to(self.dtype)
            if self.dtype != p2.dtype:
                p2 = p2.to(self.dtype)
            if self.dtype != p1_eval.dtype:
                p1_eval = p1_eval.to(self.dtype)

        return (p1, p2), p1_eval

    @property
    def inverse_10m_norm(self):
        if self.norm is not None:
            return denormalize_t(self.norm, "10")
        return None

    @property
    def inverse_20m_norm(self):
        if self.norm is not None:
            return denormalize_t(self.norm, "20")
        return None


class SR60(Dataset):
    def __init__(
        self,
        path: str,
        limit: Union[int, None] = None,
        dtype: Union[torch.dtype, None] = torch.float16,
        normalization=True,
    ):
        self.path = p.join(path, "SR60")

        if limit:
            print(f"Warning: a dataset limit of {limit} patches has been set")

        self.limit = limit

        self.norm = None
        self.norm_t = None

        if normalization:
            self.norm = norm_SR60
            self.norm_t = [
                normalize_t(self.norm, "060"),
                normalize_t(self.norm, "120"),
                normalize_t(self.norm, "360"),
                normalize_t(self.norm, "360"),
            ]
        else:
            if dtype != torch.int16:
                dtype = torch.int16
                print(
                    "Warning: if normalization is disabled, dtype is automatically set to int16"
                )

        self.dtype = dtype

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

        if self.norm_t is not None:
            p1, p2, p3, p1_eval = (
                self.norm_t[0](p1),
                self.norm_t[1](p2),
                self.norm_t[2](p3),
                self.norm_t[3](p1_eval),
            )

        if self.dtype is not None:
            if self.dtype != p1.dtype:
                p1 = p1.to(self.dtype)
            if self.dtype != p2.dtype:
                p2 = p2.to(self.dtype)
            if self.dtype != p3.dtype:
                p3 = p3.to(self.dtype)
            if self.dtype != p1_eval.dtype:
                p1_eval = p1_eval.to(self.dtype)

        return (p1, p2, p3), p1_eval

    @property
    def inverse_060m_norm(self):
        if self.norm is not None:
            return denormalize_t(self.norm, "060")
        return None

    @property
    def inverse_120m_norm(self):
        if self.norm is not None:
            return denormalize_t(self.norm, "120")
        return None

    @property
    def inverse_360m_norm(self):
        if self.norm is not None:
            return denormalize_t(self.norm, "360")
        return None


norm_SR20 = {
    "mean": {
        "10": [
            0397.72,
            0574.64,
            0507.31,
            2271.98,
        ],
        "20": [
            0891.09,
            1826.65,
            2132.83,
            2304.55,
            1538.51,
            0947.14,
        ],
    },
    "std": {
        "10": [
            0304.56,
            0354.48,
            0438.15,
            1009.62,
        ],
        "20": [
            478.28,
            782.85,
            919.56,
            984.10,
            728.89,
            589.99,
        ],
    },
}

norm_SR60 = {
    "mean": {
        "060": [
            0760.93,
            0903.02,
            0890.79,
            2436.46,
        ],
        "120": [
            1243.14,
            2033.50,
            2284.58,
            2453.61,
            1495.40,
            0962.26,
        ],
        "360": [
            0687.51,
            2439.21,
        ],
    },
    "std": {
        "060": [
            0972.88,
            0944.11,
            1034.87,
            1514.40,
        ],
        "120": [
            1060.75,
            1294.96,
            1392.86,
            1434.67,
            0979.47,
            0724.34,
        ],
        "360": [
            0901.40,
            1349.93,
        ],
    },
}
