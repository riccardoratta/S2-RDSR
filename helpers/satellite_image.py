import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import matplotlib_inline

from typing import List

from PIL import Image as PIL_image


class SatelliteImage:
    def __init__(self, path: str, max_v=20000):
        self.image = PIL_image.open(path)
        self.max_v = max_v

    @property
    def tensor(self) -> torch.Tensor:
        return (torch.clamp(F.to_tensor(self.image), 0, self.max_v) / self.max_v).type(
            torch.float16
        )


def downscale(scale: int):
    """
    Get the downscale transform, this will apply a Gaussian blur and downscale with an average
    pool.
    """
    sigma = 1 / scale
    kernel_size = 2 * (round(4 * sigma)) + 1
    return T.Compose(
        [
            T.GaussianBlur(kernel_size, sigma),
            nn.AvgPool2d(kernel_size=2),
        ]
    )


def _plot_row(patch, bands, axs):
    for i, band in enumerate(bands):
        axs[i].imshow(patch[i, :, :], cmap="Greys")
        axs[i].set_title(f"Band {band}")
        axs[i].axis("off")


def plot_patch(patch: torch.Tensor, bands: List[str]):
    """
    Plot a S2 patch with all of its bands.
    """
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

    _plot_row(patch, bands, plt.subplots(1, len(bands))[1])
