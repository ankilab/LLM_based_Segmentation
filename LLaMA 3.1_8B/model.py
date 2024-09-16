python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self._block(1, 64, 7, 3),
            nn.MaxPool2d(2),
            self._block(64, 128, 3, 1),
            nn.MaxPool2d(2),
            self._block(128, 256, 3, 1),
            nn.MaxPool2d(2),
            self._block(256, 512, 3, 1),
            nn.MaxPool2d(2),
            self._block(512, 1024, 3, 1)
        )

        self.decoder = nn.Sequential(
            self._up_block(1024, 512, 3, 1),
            self._up_block(512, 256, 3, 1),
            self._up_block(256, 128, 3, 1),
            self._up_block(128, 64, 3, 1),
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder[0](x)
        e2 = self.encoder[1:3](e1)
        e3 = self.encoder[3:5](e2)
        e4 = self.encoder[5:7](e3)
        e5 = self.encoder[7:](e4)

        d4 = self.decoder[0](torch.cat([torch.nn.functional.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True), e4], 1))
        d3 = self.decoder[1](torch.cat([torch.nn.functional.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True), e3], 1))
        d2 = self.decoder[2](torch.cat([torch.nn.functional.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True), e2], 1))
        d1 = self.decoder[3](torch.cat([torch.nn.functional.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True), e1], 1))

        return torch.sigmoid(self.final_conv(d1))

    def _block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )