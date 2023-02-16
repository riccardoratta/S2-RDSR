from typing import List, Callable, Union

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.transforms as T

from torchmetrics.metric import Metric

from helpers.satellite_image import denormalize

device = "cuda" if torch.cuda.is_available() else "cpu"


def _cuda(dataLoader):
    for batch in dataLoader:
        # inputs
        batch[0][0] = batch[0][0].to(device)
        batch[0][1] = batch[0][1].to(device)
        # target
        batch[1] = batch[1].to(device)

        yield batch


def _mean(values: List[float]):
    return torch.mean(torch.tensor(values), dtype=torch.float).item()


def model_step(
    model: nn.Module,
    dataLoader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    scaler,
    gradient_clipping: float,
):
    model.train()

    n = 0
    n_max = len(dataLoader)

    s_loss = torch.tensor(0, dtype=torch.float, device=device)

    for (x1, x2), targets in _cuda(dataLoader):
        print(f"\r> {n / n_max * 100:.2f}%", end="", flush=True)

        with torch.autocast(device_type=device, dtype=torch.float16):  # type: ignore
            loss = loss_fn(model(x1, x2), targets)

        scaler.scale(loss).backward()  #  type: ignore

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # type: ignore

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        n += 1

        s_loss += loss.detach()

    print("\r", end="", flush=True)

    return s_loss.item() / n


def model_eval(
    model: nn.Module,
    dataLoader: DataLoader,
    eval_fns: Union[Callable, List[Callable]],
):
    model.eval()

    if not isinstance(eval_fns, list):
        eval_fns = [eval_fns]

    for i, eval_fn in enumerate(eval_fns):
        if isinstance(eval_fn, Metric):
            eval_fns[i] = eval_fn.to(device)

    n = 0

    stats: List[torch.Tensor] = [
        torch.tensor(0, dtype=torch.float, device=device) for _ in eval_fns
    ]

    with torch.no_grad():
        for (x1, x2), targets in _cuda(dataLoader):

            with torch.autocast(device_type=device, dtype=torch.float16):  # type: ignore
                outputs = model(x1, x2)

            for i, eval_fn in enumerate(eval_fns):
                stats[i] += eval_fn(
                    denormalize(outputs, dtype=torch.float),
                    denormalize(targets, dtype=torch.float),
                )

            n += 1

    if len(stats) == 1:
        return (stats[0] / n).item()
    return [(stat / n).item() for stat in stats]


def at_epoch(
    model: nn.Module,
    epochs: Union[int, List[int]],
    dataLoader: DataLoader,
    eval_dataLoader: DataLoader,
    loss_fn: Callable,
    eval_fn: Union[Callable, List[Callable]],
    optimizer: Optimizer,
    scaler,
    gradient_clipping: float,
):
    """
    Train a model. Epochs could be the number of epochs to train, or a tuple with `start` and `end`
    if the model has already been trained and you wanted to restart it.

    The method should be considered as in itererator and for every step it yield the epoch number
    (starts from 1), and a tuple with the step and eval loss (i.e., training loss and validation
    loss).
    """

    if isinstance(epochs, int):
        epochs = [epochs]

    for epoch in range(*epochs):

        step_v = model_step(
            model,
            dataLoader,
            loss_fn,
            optimizer,
            scaler,
            gradient_clipping,
        )

        eval_v = model_eval(
            model,
            eval_dataLoader,
            eval_fn,
        )

        yield epoch + 1, (step_v, eval_v)


class Logger(SummaryWriter):
    """
    Wrapper class for `SummaryWriter`.
    """

    def __init__(self, name):
        super().__init__(f"runs/{name}")

    def add(self, epoch, step_v, eval_v):
        self.add_scalar("training/loss", step_v, epoch)
        self.add_scalar("validation/loss", eval_v, epoch)
