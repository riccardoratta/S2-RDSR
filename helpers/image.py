import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image as PIL_image

class Image:
    def __init__(self, path: str):
        self.image = PIL_image.open(path)

    @property
    def tensor(self) -> torch.Tensor:
        return F.to_tensor(self.image)

def downscale(scale: int):
    '''
    Get the downscale transform, this will apply a Gaussian blur and downscale with an average
    pool.
    '''
    sigma = 1 / scale; kernel_size = (2*(round(4*sigma))+1)
    return T.Compose([
        T.GaussianBlur(kernel_size, sigma), nn.AvgPool2d(kernel_size=2),
    ])
