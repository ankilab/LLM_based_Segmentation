# model.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
        ])

        self.decoder = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64),
        ])

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc_features = []
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            enc_feat = enc_features[-(i+1)]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder[i](x)

        return torch.sigmoid(self.final_conv(x))
