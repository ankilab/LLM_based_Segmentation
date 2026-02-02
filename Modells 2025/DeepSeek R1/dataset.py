import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, img_size=256, is_test=False):
        """
        Args:
            image_dir: Directory with input images
            mask_dir: Directory with mask images (defaults to image_dir if None)
            img_size: Target size for resizing
            is_test: If True, skips mask loading for inference
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.img_size = img_size
        self.is_test = is_test

        # Collect image files (png only)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and '_seg' not in f]

        # Transform for images
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Transform for masks
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('L')
        image = self.image_transform(image)

        # Handle mask if not in test mode
        mask = torch.Tensor()
        if not self.is_test:
            mask_name = f"{os.path.splitext(img_name)[0]}_m.jpg"
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
            # Binarize mask (0 or 1)
            mask = (mask > 0.5).float()

        return image, mask, img_name