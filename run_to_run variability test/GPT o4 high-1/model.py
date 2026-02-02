import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], bilinear=True):
        super().__init__()
        # encoder path
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList([
            Down(features[i], features[i+1]) for i in range(len(features)-1)
        ])
        # bottleneck
        prev_channels = features[-1]
        bottleneck_channels = prev_channels * 2
        self.bottleneck = DoubleConv(prev_channels, bottleneck_channels)
        # decoder path
        enc_channels = features  # [64,128,256,512]
        self.ups = nn.ModuleList()
        in_ch = bottleneck_channels
        for ch in reversed(enc_channels):
            self.ups.append(Up(in_ch + ch, ch, bilinear))
            in_ch = ch
        # final 1x1 conv
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encode
        encs = [self.inc(x)]
        for down in self.downs:
            encs.append(down(encs[-1]))
        # bottleneck
        x = self.bottleneck(encs[-1])
        # decode with skip connections
        for i, up in enumerate(self.ups):
            skip = encs[-1 - i]
            x = up(x, skip)
        x = self.outc(x)
        return torch.sigmoid(x)
