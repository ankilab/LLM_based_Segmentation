import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up1 = DoubleConv(256 + 512, 256)
        self.up2 = DoubleConv(128 + 256, 128)
        self.up3 = DoubleConv(64 + 128, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)

        u1 = self.up1(torch.cat([d3, F.interpolate(d4, d3.size()[2:])], 1))
        u2 = self.up2(torch.cat([d2, F.interpolate(u1, d2.size()[2:])], 1))
        u3 = self.up3(torch.cat([d1, F.interpolate(u2, d1.size()[2:])], 1))
        return self.final_conv(u3)
