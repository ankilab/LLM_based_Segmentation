# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """
    Upscaling then double conv. If bilinear, use Upsample + 1×1 conv to halve channels;
    otherwise use ConvTranspose2d which halves channels by construction.
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # reduce channels by 2×
            self.channel_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
            in_ch = in_ch // 2
        else:
            # this ConvTranspose2d will halve channels automatically
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            in_ch = in_ch // 2

        # after up and (optional) reduce, we'll concat with the skip-connection
        # which has the same number of channels = in_ch
        self.conv = DoubleConv(in_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # if we have the extra 1×1 conv, apply it
        if hasattr(self, 'channel_reduce'):
            x1 = self.channel_reduce(x1)

        # pad to match spatial dims
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1   = Up(1024, 512, bilinear)
        self.up2   = Up(512, 256, bilinear)
        self.up3   = Up(256, 128, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # 64 channels
        x2 = self.down1(x1)    # 128
        x3 = self.down2(x2)    # 256
        x4 = self.down3(x3)    # 512
        x5 = self.down4(x4)    # 1024
        x  = self.up1(x5, x4)  # back to 512
        x  = self.up2(x,  x3)  # 256
        x  = self.up3(x,  x2)  # 128
        x  = self.up4(x,  x1)  #  64
        logits = self.outc(x)
        return torch.sigmoid(logits)
