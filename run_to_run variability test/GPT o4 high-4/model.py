# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # when using convtranspose, in_ch should be twice out_ch
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad to match
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024], bilinear=True):
        super().__init__()
        # Encoder path
        self.inc   = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])
        # Bottleneck
        self.bottleneck = DoubleConv(features[4], features[4]*2)
        # Decoder path (reverse order)
        self.up4 = Up(features[4]*2 + features[4], features[4], bilinear)
        self.up3 = Up(features[4]   + features[3], features[3], bilinear)
        self.up2 = Up(features[3]   + features[2], features[2], bilinear)
        self.up1 = Up(features[2]   + features[1], features[1], bilinear)
        self.up0 = Up(features[1]   + features[0], features[0], bilinear)
        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)       # 256→256
        x1 = self.down1(x0)    # 256→128
        x2 = self.down2(x1)    # 128→ 64
        x3 = self.down3(x2)    #  64→ 32
        x4 = self.down4(x3)    #  32→ 16
        x5 = self.bottleneck(x4)  # 16→16
        # Upsample path
        x  = self.up4(x5, x4)  # 16→32
        x  = self.up3(x,  x3)  # 32→64
        x  = self.up2(x,  x2)  # 64→128
        x  = self.up1(x,  x1)  #128→256
        x  = self.up0(x,  x0)  #256→256
        return self.final(x)
