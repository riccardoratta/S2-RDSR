import torch
import torch.nn as nn

import torchvision.transforms as T

class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def forward(self, x1, x2):
        
        return T.Resize(x1.size(dim=3), T.InterpolationMode.BICUBIC)(x2)
