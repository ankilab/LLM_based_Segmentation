# unet_segmentation/dataset.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix="_m", target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.suffix = suffix
        self.target_size = target_size

        self.image_paths = sorted([p for p in glob.glob(os.path.join(self.image_dir, "*.jpg")) if self.suffix not in os.path.basename(p)])

        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        mask_name = image_name.replace(".jpg", f"{self.suffix}.jpg")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Ensure binary mask

        return image, mask, image_name
