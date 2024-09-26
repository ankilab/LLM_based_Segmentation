import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder = nn.ModuleList([
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        ])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.up_convs = nn.ModuleList([
            up_conv(1024, 512),
            up_conv(512, 256),
            up_conv(256, 128),
            up_conv(128, 64),
        ])

        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
        ])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []

        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            x = torch.cat([x, encoder_outputs[-(i + 1)]], dim=1)
            x = self.decoder[i](x)

        return torch.sigmoid(self.final_conv(x))
