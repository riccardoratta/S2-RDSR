import os
import os.path as p
import json
import argparse

import torch

from helpers.sentinel import Bands

from typing import List, Dict, Callable

from helpers.log import error
from helpers.sentinel import Bands, Resolution
from helpers.dataset.SR20 import SR20

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Compute the standard devation and mean of a SR dataset."
)

parser.add_argument(
    "path",
    metavar="P",
    help="Path to a SR dataset",
)

parser.add_argument(
    "--type",
    choices=["20", "60"],
    nargs="+",
    required=True,
    help='Specify the dataset type, could be "20" or "60" respectively for SR 20m or 60m dataset',
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    help="Specify batch size for the data loader",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Specify how many subprocesses to use for data loading.",
)


def batch_band(batch: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]):
    """
    Compute a function `func` over the second dimension (channel) of batch of tensors.
    """
    return torch.cat(
        [torch.unsqueeze(func(batch[:, i]), 0) for i in range(batch.shape[1])], 0
    )


def tensor_JSONEncoder(value: torch.Tensor) -> List[float]:
    return value.tolist()


def dataset_std_mean(dataLoader: DataLoader, resolutions: List[str]):
    std: Dict[str, torch.Tensor] = {}

    for r in resolutions:
        std[r] = torch.zeros(len(Bands[f"m{r}"].value))
    mean = std.copy()

    n = 0
    n_max = len(dataLoader)
    limit = 0
    for value, _ in dataLoader:
        print(f"\r{n / n_max * 100:.2f}%", end="", flush=True)

        for i, r in enumerate(resolutions):
            std[r] += batch_band(value[i], torch.std)
            mean[r] += batch_band(value[i], torch.mean)

        limit += 1
        n += 1

        if limit >= 100:
            # print(f'Reset running std and mean at {limit} steps')
            for m in std:
                std[m] /= limit
                mean[m] /= limit
            limit = 0

    print("\r", end="")

    for m in std:
        std[m] /= limit
        mean[m] /= limit

    return std, mean


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        error(f"Input path {args.path} is not valid.")

    if args.num_workers != 2:
        print(f"Numer of workers for data loading: {args.num_workers}")

    if "20" in args.type:
        dataset = SR20(args.path)

        print(f"Computing SR20 mean and std over {len(dataset)} patches..")

        std, mean = dataset_std_mean(
            DataLoader(dataset, args.batch_size, num_workers=args.num_workers),
            Resolution.SR20.value,
        )

        with open(p.join(args.path, "SR20", "norm.json"), "w") as f:
            f.write(
                json.dumps(
                    {"mean": mean, "std": std}, default=tensor_JSONEncoder, indent=2
                )
            )

        print("SR20 completed!")

    if "60" in args.type:
        raise NotImplementedError()
