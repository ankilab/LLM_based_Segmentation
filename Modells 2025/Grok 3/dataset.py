import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, image_suffix="", mask_suffix="_seg", transform=None):
        """
        Custom Dataset for loading grayscale images and corresponding binary masks.
        mask_dir can be same as image_dir if masks are in the same folder.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform

        # Get list of image files (only .png)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')
                                   and not f.endswith(f"{self.mask_suffix}.png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Construct mask name
        base_name = img_name.split('.')[0]
        mask_name = f"{base_name}{self.mask_suffix}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask as grayscale
        image = Image.open(img_path).convert('L')  # Grayscale
        mask = Image.open(mask_path).convert('L')  # Grayscale (binary mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure binary mask (0 or 1)
        mask = (mask > 0.5).float()
        return image, mask, img_name