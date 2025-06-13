# unet_segmentation/model.py

import torch
import torch.nn as nn
import torchinfo

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3 = double_conv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = double_conv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1 = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        x = self.down4(x)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)

        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)

        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)

        return torch.sigmoid(self.final_conv(x))

def print_model_summary(model, input_size=(1, 256, 256)):
    summary = torchinfo.summary(model, input_size=input_size, verbose=0)
    print(summary)
    print(f"Total trainable parameters: {summary.trainable_params}")