# unet_segmentation/model.py

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        up4 = self.conv4(merge4)

        up3 = self.up3(up4)
        merge3 = torch.cat([up3, d3], dim=1)
        up3 = self.conv3(merge3)

        up2 = self.up2(up3)
        merge2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(merge2)

        up1 = self.up1(up2)
        merge1 = torch.cat([up1, d1], dim=1)
        up1 = self.conv1(merge1)

        return self.sigmoid(self.final(up1))
