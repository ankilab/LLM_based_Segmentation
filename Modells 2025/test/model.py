# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_c, out_c)
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        self.conv  = DoubleConv(in_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diff = [s2 - s1 for s1, s2 in zip(x.shape[2:], skip.shape[2:])]
        x = F.pad(x, [d//2 for d in diff[::-1]] * 2)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256]):
        super().__init__()
        self.inc = DoubleConv(in_ch, features[0])
        self.downs = nn.ModuleList([
            Down(features[i], features[i+1]) for i in range(len(features)-1)
        ])
        self.ups   = nn.ModuleList([
            Up(features[i+1], features[i]) for i in reversed(range(len(features)-1))
        ])
        self.outc  = nn.Conv3d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        for d in self.downs:
            skips.append(x)
            x = d(x)
        for u, skip in zip(self.ups, reversed(skips)):
            x = u(x, skip)
        return self.outc(x)
