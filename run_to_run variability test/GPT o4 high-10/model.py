# unet_segmentation/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(prev_ch, f))
            prev_ch = f

        # Bottleneck
        bottleneck_ch = prev_ch * 2
        self.bottleneck = DoubleConv(prev_ch, bottleneck_ch)

        # Decoder
        self.ups = nn.ModuleList()
        curr_ch = bottleneck_ch
        for f in reversed(features):
            # up-conv reduces channels from curr_ch → f
            self.ups.append(nn.ConvTranspose2d(curr_ch, f, kernel_size=2, stride=2))
            # after concat, channels = f (skip) + f (upsampled) = 2f
            self.ups.append(DoubleConv(in_ch=2*f, out_ch=f))
            curr_ch = f

        # Final 1×1 conv to map to desired out_channels
        self.final_conv = nn.Conv2d(curr_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            up_conv   = self.ups[idx]
            double_conv = self.ups[idx+1]

            x = up_conv(x)
            skip = skip_connections[idx // 2]

            # in case rounding made sizes off by 1
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = double_conv(x)

        return self.final_conv(x)
