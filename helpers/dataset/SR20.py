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
        dtype=torch.float16,
    ):
        self.path = p.join(path, "SR20")

        if limit:
            print("Warning: a dataset limit has been set")

        self.limit = limit

        self.dtype = dtype

        self.transforms = None

        try:
            with open(p.join(self.path, "norm.json")) as file:
                norm = json.load(file)
                self.transforms = [
                    T.Normalize(mean=norm["mean"]["10"], std=norm["std"]["10"]),
                    T.Normalize(mean=norm["mean"]["20"], std=norm["std"]["20"]),
                    T.Normalize(mean=norm["mean"]["20"], std=norm["std"]["20"]),
                ]
        except FileNotFoundError:
            print("Warning: 'norm.json' is missing from the dataset")

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

        if self.dtype != torch.float:
            p1, p2, p1_eval = (
                p1.to(self.dtype),
                p2.to(self.dtype),
                p1_eval.to(self.dtype),
            )

        if self.transforms:
            p1, p2, p1_eval = (
                self.transforms[0](p1),
                self.transforms[1](p2),
                self.transforms[2](p1_eval),
            )

        return (p1, p2), p1_eval

    @property
    def inverse_10m_norm(self):
        if self.transforms is not None:
            return _get_inverse_norm(self.transforms[0])
        return None

    @property
    def inverse_20m_norm(self):
        if self.transforms is not None:
            return _get_inverse_norm(self.transforms[1])
        return None


def _get_inverse_norm(v):
    return T.Normalize(
        mean=-(torch.tensor(v.mean) / torch.tensor(v.std)), std=1 / torch.tensor(v.std)
    )


"""
def get_sr20_dataLoaders(dataset, batch_size, transforms=None, limit=None, split=[0.9, 0.1], workers=2):
    
    if len(dataset) != 0 and not dataset.endswith('/'):
        dataset += '/'

    if not os.path.exists(f'data/{dataset}'):
        raise FileNotFoundError(f"dataset at '{os.getcwd()}/data/{dataset}' doesn't exists")

    if transforms is not None:
        assert len(transforms) == 3, 'please specify a transformer for every patch (1x, 2x, 1x_eval)'

    if split is not None:
        data, data_eval, = random_split(
            dataset_sr20(f'data/{dataset}patches', transforms, limit), [0.9, 0.1])

        return (
            DataLoader(data, batch_size, shuffle=True, num_workers=workers),
            DataLoader(data_eval, batch_size, shuffle=False, num_workers=workers)
        )
    else:
        return DataLoader(dataset_sr20(f'data/{dataset}patches', transforms, limit),
                          batch_size, num_workers=2)

def _get_zfill(path):
    return len(os.path.basename(
        glob.glob(f'{path}/sr20/20/*.pt')[0]).replace('.pt', ''))
"""
