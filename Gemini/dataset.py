import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        # Filter out only the images and masks based on the naming conventions
        self.image_paths = [f for f in os.listdir(data_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.mask_paths = [f.replace('.png', '_seg.png') for f in self.image_paths]  # Generate corresponding mask names

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        mask_path = os.path.join(self.data_dir, self.mask_paths[idx])

        # Check if the mask exists, otherwise raise an error
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(image_path).convert('L')  # Ensure grayscale
        mask = Image.open(mask_path).convert('L')  # Ensure binary mask

        if self.transform is not None:
            image = self.transform(image)  # Apply transform only to the image
            mask = self.transform(mask)

        # mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # Convert and add channel dimension

        return image, mask

