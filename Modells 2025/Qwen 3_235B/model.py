import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.bottleneck = CBR(512, 1024)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.enc1(x)
        x = self.pool(conv1)
        conv2 = self.enc2(x)
        x = self.pool(conv2)
        conv3 = self.enc3(x)
        x = self.pool(conv3)
        conv4 = self.enc4(x)
        x = self.pool(conv4)

        x = self.bottleneck(x)

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)