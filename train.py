import os.path as p
import argparse

from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

import torchvision.transforms as T

import torchmetrics

from models.RDSR import RDSR

from helpers.log import error
from helpers.model import at_epoch
from helpers.dataset.SR20 import SR20

parser = argparse.ArgumentParser(
    description="Train a RDSR deep neural network",
)

parser.add_argument(
    "path",
    metavar="P",
    help="Path to a SR dataset",
)

parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    help="Number of epochs to train the model",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for the training",
)

parser.add_argument(
    "--model",
    type=str,
    help="Path to the pre-trained model",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="How many subprocesses to use for data loading",
)


def _get_optimizer(model: torch.nn.Module, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)


def _get_scheduler(optimizer: optim.Optimizer, learning_rate):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=learning_rate)


if __name__ == "__main__":
    args = parser.parse_args()

    start = time()

    if not p.isdir(args.path):
        error(f"Input path {args.path} is not valid.")

    if args.model is not None:
        if not p.isfile(args.model):
            error(f"Model path {args.path} is not valid.")
        # TODO: implement model retrival

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("Warning: no GPU detected, traning can be very slow")

    if args.num_workers != 2:
        print(f"Numer of workers for data loading: {args.num_workers}")

    dataset = SR20(args.path)

    sub_dataset, eval_sub_dataset = random_split(dataset, [0.9, 0.1])

    dataLoader, eval_dataLoader = (
        DataLoader(
            sub_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(False if device == "cpu" else True),
        ),
        DataLoader(
            eval_sub_dataset,
            args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(False if device == "cpu" else True),
        ),
    )

    learning_rate = 0.0002

    model = RDSR().to(device)

    optimizer = _get_optimizer(model, learning_rate)
    scheduler = _get_scheduler(optimizer, learning_rate)

    print("Training epoch n. 001..")

    for epoch, (step_v, eval_v) in at_epoch(
        model,
        args.epochs,
        dataLoader,
        eval_dataLoader,
        nn.L1Loss(),
        torchmetrics.MeanSquaredError(),
        optimizer,
        scheduler,
        # gradient_clipping=1,
        inverse_norm=dataset.inverse_20m_norm,
    ):
        print(f"Epoch {epoch:03d}/{args.epochs:03d} completed.")

        print(f"> {step_v:.3f}, {eval_v:.3f}")

        if epoch != args.epochs:
            print(f"Training epoch n. {epoch+1:03d}..")

    print(f"Training completed in {time() - start:.2f}s!")

    torch.save(
        {
            "model": model.state_dict(),
            "epochs": args.epochs,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        "model.pt",
    )
