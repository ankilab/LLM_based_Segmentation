import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()
        # Define contracting (encoding) path
        self.encoder = nn.Sequential(
            self._contracting_block(in_channels, 32),
            self._contracting_block(32, 64),
            self._contracting_block(64, 128),
            self._contracting_block(128, 256),
            self._contracting_block(256, 512),
        )

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Define expanding (decoding) path
        self.decoder = nn.Sequential(
            self._expanding_block(512, 256),
            self._expanding_block(256, 128),
            self._expanding_block(128, 64),
            self._expanding_block(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def _contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return block

    def _expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encode the input image
        x = self.encoder(x)

        # Bottleneck layer
        x = self.bottleneck(x)

        # Decode the features
        x = self.decoder(x)

        return x