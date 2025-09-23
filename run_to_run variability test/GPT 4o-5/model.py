import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    A sequence of two convolutional layers each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary segmentation (in_channels=1, out_channels=1).
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: double conv → pool
        for feature in features:
            self.encoder_layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: upconv → double conv
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        reversed_features = features[::-1]
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(DoubleConv(feature * 2, feature))

        # Final 1x1 conv to get to out_channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for enc in self.encoder_layers:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for decoder

        # Decoder path
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            # In case input sizes are not perfectly divisible, crop
            if x.shape != skip_connection.shape:
                x = self._crop_to(x, skip_connection.shape[2], skip_connection.shape[3])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_layers[idx](x)

        return self.final_conv(x)

    @staticmethod
    def _crop_to(tensor, target_h, target_w):
        """
        Center-crop `tensor` to (target_h, target_w).
        """
        _, _, h, w = tensor.size()
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return tensor[:, :, dh:dh + target_h, dw:dw + target_w]
