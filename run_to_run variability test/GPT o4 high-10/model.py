# unet_segmentation/model.py
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

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2,2)
        prev_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(prev_ch, f))
            prev_ch = f
        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, prev_ch*2)
        # Decoder
        self.ups = nn.ModuleList()
        feats_rev = features[::-1]
        for f in feats_rev:
            self.ups.append(nn.ConvTranspose2d(prev_ch*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(prev_ch*2, f))
            prev_ch = f
        # Final
        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)
