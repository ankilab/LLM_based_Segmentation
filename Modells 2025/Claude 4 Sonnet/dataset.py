import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir: str = None, mask_suffix='', transform=None, image_size=(256, 256)):
        """
        Dataset for loading grayscale images and corresponding binary masks.

        Args:
            image_dir (str): Directory containing images and masks
            mask_suffix (str): Suffix for mask files (e.g., '_seg' for 'image_seg.png')
            transform (callable, optional): Optional transform to be applied
            image_size (tuple): Target size for images and masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.image_size = image_size

        # Get all PNG files
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

        # Filter out mask files and keep only original images
        self.image_files = []
        for f in all_files:
            if self.mask_suffix and f.replace('.png', '').endswith(mask_suffix):
                continue
                # Check if corresponding mask exists
            mask_name = f.replace('.png', f'{mask_suffix}.png')
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                self.image_files.append(f)
                # if mask_name in all_files:
                #     # look for the mask in mask_dir not only in image_dir
                #     if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                #         self.image_files.append(f)

        self.image_files.sort()

        # Define transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        print(f"Found {len(self.image_files)} image-mask pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load mask
        mask_name = img_name.replace('.png', f'{self.mask_suffix}.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Open and convert images
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask, img_name