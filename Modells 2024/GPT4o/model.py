import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.ModuleList([
            conv_block(1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024)
        ])

        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        encoder_features = []
        for enc in self.encoder:
            x = enc(x)
            encoder_features.append(x)
            x = self.maxpool(x)

        # Start decoding
        for i, dec in enumerate(self.decoder):
            x = self.upconv[i](x)
            enc_feature = encoder_features[-(i + 2)]
            # Crop encoder features to match the size of x before concatenation
            enc_feature = self.crop(enc_feature, x)
            x = torch.cat([x, enc_feature], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final_conv(x))

    def crop(self, enc_feature, x):
        """
        Crop the encoder feature map to match the size of the decoder feature map.
        Args:
            enc_feature: feature map from the encoder path.
            x: feature map from the decoder path (upsampled).
        """
        _, _, h, w = x.size()
        enc_feature = enc_feature[:, :, :h, :w]
        return enc_feature
