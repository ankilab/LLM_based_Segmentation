# unet_segmentation/model.py

import torch
import torch.nn as nn

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

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        # Encoder
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        # Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Decoder
        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(2*f, f, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(2*f, f))
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            trans = self.ups[idx]
            conv  = self.ups[idx+1]
            x = trans(x)
            skip = skip_connections[idx//2]
            # pad if needed
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = conv(x)
        return self.final(x)
