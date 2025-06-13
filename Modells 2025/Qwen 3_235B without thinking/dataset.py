# unet_segmentation/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import glob

class GrayscaleBinarySegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, suffix="_m"):
        """
        Args:
            image_dir (str): Path to grayscale images.
            mask_dir (str): Path to masks. If None, assumes same directory with suffix.
            transform (callable, optional): Optional transform to be applied on a sample.
            suffix (str): Suffix used for mask files.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.transform = transform
        self.suffix = suffix

        # Only include .png files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_paths = [os.path.join(image_dir, f) for f in self.image_files]

        # Match mask names
        self.mask_paths = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}{suffix}.png"
            mask_path = os.path.join(self.mask_dir, mask_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file {mask_path} not found.")
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        mask = Image.open(self.mask_paths[idx]).convert("L")  # binary mask

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)  # binary

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # add channel dim
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask