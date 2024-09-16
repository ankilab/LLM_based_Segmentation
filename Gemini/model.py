import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.decoder = nn.ModuleList()
        for feature in reversed(features[1:]):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=3, stride=2, padding=1))
            self.decoder.append(DoubleConv(feature * 2, feature))
        self.decoder.append(nn.ConvTranspose2d(features[0], out_channels, kernel_size=3, stride=2, padding=1))
        self.final = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        for decoder, skip in zip(self.decoder[::2], reversed(skip_connections)):
            x = decoder(x)
            x = torch.cat((x, skip), dim=1)
            x = self.decoder[1:][::2](x)

        x = self.final(x)
        return x