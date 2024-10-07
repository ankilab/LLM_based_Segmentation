import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._make_conv_block(in_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)
        self.enc4 = self._make_conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._make_conv_block_no_pooling(512, 1024)

        # Decoder
        self.dec4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final Layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Define conv_block and conv_block_no_pooling methods
        self.conv_block = self._make_conv_block
        self.conv_block_no_pooling = self._make_conv_block_no_pooling

    def _make_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _make_conv_block_no_pooling(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with resizing and skip connections
        d4 = self.dec4(b)
        d4 = self.conv_block_no_pooling(512, 512)(d4)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.conv_block_no_pooling(1024, 512)(d4)  # Apply conv block AFTER concatenation and resizing

        d3 = self.dec3(d4)
        d3 = self.conv_block_no_pooling(512, 256)(d3)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((e3, d3), dim=1)

        d2 = self.dec2(d3)
        d2 = self.conv_block_no_pooling(256, 128)(d2)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((e2, d2), dim=1)

        d1 = self.dec1(d2)
        d1 = self.conv_block_no_pooling(128, 64)(d1)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((e1, d1), dim=1)


        # Final Layer
        out = self.final(d1)
        return out