import math

import torch

from typing import Union

import torchvision.transforms.functional as F


Shape = Union[torch.Size, list[int]]


def _get_pad(shape: Shape, d: int, inset: int, border: int):
    return (math.ceil(shape[d] / inset) * inset + border) - shape[d]


def _get_all_pad(shape: Shape, inset: int, border: int, mult=1):
    pad_h = _get_pad(shape, 1, inset, border)
    pad_w = _get_pad(shape, 2, inset, border)

    return list(
        (
            mult * _f(pad_w / 2),
            mult * _f(pad_h / 2),
            mult * _c(pad_w / 2),
            mult * _c(pad_h / 2),
        )
    )


def to_batch(
    image: torch.Tensor,
    patch_size: int,
    border: int,
    batch_size: int,
):
    inset = patch_size - border

    pad_image = F.pad(
        image,
        _get_all_pad(image.shape, inset, border),
        padding_mode="reflect",
    )

    n = 0
    buffer = []
    for i in range(0, image.shape[1], inset):
        for j in range(0, image.shape[2], inset):
            buffer.append(pad_image[:, i : i + patch_size, j : j + patch_size])
            n += 1
            if n >= batch_size:
                yield torch.stack(buffer, 0)
                n = 0
                buffer = []

    if len(buffer) != 0:
        yield torch.stack(buffer, 0)


def to_image(
    batches: torch.Tensor,
    patch_size: int,
    border: int,
    original_shape: Shape,
):
    inset = patch_size - border

    pad_image = F.pad(
        torch.zeros(
            original_shape,
            dtype=batches[0].dtype,
        ),
        _get_all_pad(original_shape, inset, border),
    )

    n = 0
    for i in range(0, original_shape[1], inset):
        for j in range(0, original_shape[2], inset):
            pad_image[:, i : i + patch_size, j : j + patch_size] = batches[n]
            n += 1

    return F.pad(pad_image, _get_all_pad(original_shape, inset, border, -1))


def _f(v: float):
    return math.floor(v)


def _c(v: float):
    return math.ceil(v)
