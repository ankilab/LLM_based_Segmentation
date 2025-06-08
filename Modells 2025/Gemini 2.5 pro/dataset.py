# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class UNetDataset(Dataset):
    """
    Custom Dataset for loading grayscale images and their corresponding binary masks.
    """

    def __init__(self, image_dir, mask_dir, image_filenames, mask_suffix="_seg", transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            image_filenames (list): List of image filenames to load.
            mask_suffix (str): Suffix of the mask files.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        #self.image_filenames = image_filenames
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mask_transform = mask_transform

        # ── keep only true images, drop any *_seg.png that slipped in ──
        self.image_filenames = [
            f for f in image_filenames
            if not f.lower().endswith(f"{mask_suffix}.png")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        stem, _ = os.path.splitext(img_name)
        mask_name = f"{stem}{self.mask_suffix}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Open image and mask
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0.5).float()

        return image, mask, img_name