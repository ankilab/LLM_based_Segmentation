import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.mp_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD = x2.size(2)-x1.size(2)
        diffH = x2.size(3)-x1.size(3)
        diffW = x2.size(4)-x1.size(4)
        x1 = F.pad(x1, [
            diffW//2, diffW-diffW//2,
            diffH//2, diffH-diffH//2,
            diffD//2, diffD-diffD//2
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1   = Up(512+512, 256)
        self.up2   = Up(256+256, 128)
        self.up3   = Up(128+128, 64)
        self.up4   = Up(64+64,   64)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)
