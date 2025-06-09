import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()
        # Define contracting (encoding) path
        self.enc1 = self._contracting_block(in_channels, 32)
        self.enc2 = self._contracting_block(32, 64)
        self.enc3 = self._contracting_block(64, 128)
        self.enc4 = self._contracting_block(128, 256)
        self.enc5 = self._contracting_block(256, 512)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Define expanding (decoding) path
        self.upconv5 = self._expanding_block(1024, 512)
        self.upconv4 = self._expanding_block(512 + 512, 256)
        self.upconv3 = self._expanding_block(256 + 256, 128)
        self.upconv2 = self._expanding_block(128 + 128, 64)
        self.upconv1 = self._expanding_block(64 + 64, 32)

        # Final layer
        self.final_conv = nn.Conv2d(32 + 32, out_channels, kernel_size=3, padding=1)

    def _contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return block

    def _expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    import torch.nn.functional as F

    def forward(self, x):
        # Encoder path
        enc1_out = self.enc1(x)  # 32 channels
        enc2_out = self.enc2(enc1_out)  # 64 channels
        enc3_out = self.enc3(enc2_out)  # 128 channels
        enc4_out = self.enc4(enc3_out)  # 256 channels
        enc5_out = self.enc5(enc4_out)  # 512 channels

        # Bottleneck
        bottleneck_out = self.bottleneck(enc5_out)  # 1024 channels

        # Decoder path with skip connections
        dec5_out = self.upconv5(bottleneck_out)  # Up to 512 channels
        dec4_out = self.upconv4(torch.cat([enc5_out, self.crop_tensor(enc5_out, dec5_out)], dim=1))  # Skip connection
        dec3_out = self.upconv3(torch.cat([enc4_out, self.crop_tensor(enc4_out, dec4_out)], dim=1))  # Skip connection
        dec2_out = self.upconv2(torch.cat([enc3_out, self.crop_tensor(enc3_out, dec3_out)], dim=1))  # Skip connection
        dec1_out = self.upconv1(torch.cat([enc2_out, self.crop_tensor(enc2_out, dec2_out)], dim=1))  # Skip connection

        final_out = self.final_conv(
            torch.cat([enc1_out, self.crop_tensor(enc1_out, dec1_out)], dim=1))  # Final skip connection

        # Upsample the final output to match the original input size (256x256)
        final_out = F.interpolate(final_out, size=(256, 256), mode='bilinear', align_corners=False)

        return final_out

    def crop_tensor(self, enc_out, dec_out):
        """Center-crop decoder output to match the size of encoder output."""
        enc_size = enc_out.size()[2:]  # (H, W) of encoder output
        dec_size = dec_out.size()[2:]  # (H, W) of decoder output

        # Calculate cropping sizes
        crop_h = (dec_size[0] - enc_size[0]) // 2
        crop_w = (dec_size[1] - enc_size[1]) // 2

        # Crop decoder output to match encoder size
        dec_out = dec_out[:, :, crop_h:crop_h + enc_size[0], crop_w:crop_w + enc_size[1]]

        return dec_out

