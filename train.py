import os.path as p
import argparse

from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

import torchvision.transforms as T

from models.RDSR import RDSR

from helpers.log import error
from helpers.model import at_epoch, Logger
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
    "--model-out",
    type=str,
    default="model.out.pt",
    help="Path to the trained model, updated every epoch",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
    help="How many subprocesses to use for data loading",
)

parser.add_argument(
    "--tensorboard",
    type=bool,
    default=True,
    help="If to use TensorBoard to monitor training",
)


def _rmse_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(x, y))


if __name__ == "__main__":
    args = parser.parse_args()

    start = time()

    if not p.isdir(args.path):
        error(f"Input path {args.path} is not valid.")

    if args.model == args.model_out:
        error("Input model must be different from output model")

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

    state = {
        "model": RDSR().to(device),
    }

    state = {
        **state,
        "epoch": 0,
        "optimizer": optim.Adam(
            state["model"].parameters(),
            lr=learning_rate,
        ),
        "scaler": torch.cuda.amp.GradScaler(),  # type: ignore
    }

    if args.model is not None:
        if not p.isfile(args.model):
            error(f"Model path {args.model} is not valid.")

        loader = torch.load(args.model, map_location=device)

        for k in loader:
            if hasattr(state[k], "load_state_dict"):
                state[k].load_state_dict(loader[k])
            else:
                state[k] = loader[k]

        print("State loader")

    print(f"Training epoch n. {state['epoch']+1:03d}..")

    logger = None

    if args.tensorboard:
        logger = Logger("RDSR")

    for epoch, (step_v, eval_v) in at_epoch(
        state["model"],
        [state["epoch"], state["epoch"] + args.epochs],
        dataLoader,
        eval_dataLoader,
        nn.L1Loss(),
        _rmse_fn,
        state["optimizer"],
        state["scaler"],
        gradient_clipping=1,
    ):
        print(f"Epoch {epoch:03d}/{args.epochs:03d} completed.")

        print(f"> {step_v:.3f}, {eval_v:.3f}")

        if logger is not None:
            logger.add(epoch, step_v, eval_v)

        state["epoch"] = epoch

        torch.save(
            {
                k: (value.state_dict() if hasattr(value, "state_dict") else value)
                for k, value in state.items()
            },
            args.model_out,
        )

        if epoch != args.epochs:
            print(f"Training epoch n. {epoch+1:03d}..")

    print(f"Training completed in {time() - start:.2f}s!")
