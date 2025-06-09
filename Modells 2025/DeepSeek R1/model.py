import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with pooling
        e1 = self.enc1(x)  # [b, 64, 256, 256]
        e2 = self.enc2(self.pool(e1))  # [b, 128, 128, 128]
        e3 = self.enc3(self.pool(e2))  # [b, 256, 64, 64]
        e4 = self.enc4(self.pool(e3))  # [b, 512, 32, 32]

        # Bottleneck (with additional pooling)
        b = self.bottleneck(self.pool(e4))  # [b, 1024, 16, 16]

        # Decoder with skip connections
        d4 = self.upconv4(b)  # [b, 512, 32, 32]
        d4 = torch.cat([d4, e4], dim=1)  # [b, 1024, 32, 32]
        d4 = self.dec4(d4)  # [b, 512, 32, 32]

        d3 = self.upconv3(d4)  # [b, 256, 64, 64]
        d3 = torch.cat([d3, e3], dim=1)  # [b, 512, 64, 64]
        d3 = self.dec3(d3)  # [b, 256, 64, 64]

        d2 = self.upconv2(d3)  # [b, 128, 128, 128]
        d2 = torch.cat([d2, e2], dim=1)  # [b, 256, 128, 128]
        d2 = self.dec2(d2)  # [b, 128, 128, 128]

        d1 = self.upconv1(d2)  # [b, 64, 256, 256]
        d1 = torch.cat([d1, e1], dim=1)  # [b, 128, 256, 256]
        d1 = self.dec1(d1)  # [b, 64, 256, 256]

        return self.out(d1)  # [b, 1, 256, 256]