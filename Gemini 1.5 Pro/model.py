# model.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block_no_pooling(512, 1024)


        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)


        # Final Layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # Padding added here
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),  # Padding added here
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def conv_block_no_pooling(self, in_c, out_c): #helper function for cleaner code
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # Padding added here
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),  # Padding added here
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            self.conv_block_no_pooling(out_c, out_c)
        )


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d1 = self.dec1(d2)
        d1 = torch.cat((e1, d1), dim=1)

        # Final Layer
        out = self.final(d1)
        return out