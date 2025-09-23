# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.ups = nn.ModuleList()
        rev_feats = features[::-1]
        for feat in rev_feats:
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat*2, feat))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            upconv = self.ups[idx]
            double_conv = self.ups[idx+1]
            x = upconv(x)
            skip = skip_connections[idx//2]
            # pad if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = double_conv(x)

        return torch.sigmoid(self.final_conv(x))
