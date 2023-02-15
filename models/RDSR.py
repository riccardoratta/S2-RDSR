import torch
import torch.nn as nn

import torchvision.transforms as T

class _Dense(nn.Module):
    def __init__(self, k0: int, k: int):
        super(_Dense, self).__init__()
        
        self.act = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(k0,     k,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k0+k,   k,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(k0+2*k, k,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(k0+3*k, k0, kernel_size=3, padding=1)
        
    def forward(self, x0: torch.Tensor):
        
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(torch.cat((x0, x1),     dim=1)))
        x3 = self.act(self.conv3(torch.cat((x0, x1, x2), dim=1)))
        
        return self.conv4(torch.cat((x0, x1, x2, x3), dim=1))
    
class _RRD(nn.Module):
    def __init__(self, scaler: float):
        super(_RRD, self).__init__()
        
        self.scaler = scaler
        
        self.dense1 = _Dense(k0=128, k=64)
        self.dense2 = _Dense(k0=128, k=64)
        self.dense3 = _Dense(k0=128, k=64)
        
        
    def forward(self, x0: torch.Tensor):
        
        x1 = x0 + self.dense1(x0) * self.scaler
        x2 = x1 + self.dense2(x1) * self.scaler
        x3 = x2 + self.dense3(x2) * self.scaler
        
        return x3 + x0

class RDSR(nn.Sequential):
    def __init__(
        self, in_channels: int = 10, out_channels: int = 6, scaler: float = 0.2, rrd_n: int = 3):

        super(RDSR, self).__init__(
            nn.Conv2d(in_channels, 128            , kernel_size=3, padding=1),
            *[_RRD(scaler) for _ in range(rrd_n)],
            nn.Conv2d(126        ,    out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        
        x2 = T.Resize(x1.size(dim=2), interpolation=T.InterpolationMode.BICUBIC)(x2)
   
        return self(torch.cat((x1, x2), dim=1)) + x2