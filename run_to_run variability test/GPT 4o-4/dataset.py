# unet_segmentation/dataset.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix="_m", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.suffix = suffix
        self.transform = transform

        self.image_paths = sorted([p for p in glob.glob(os.path.join(self.image_dir, "*.jpg")) if self.suffix not in os.path.basename(p)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        mask_name = image_name.replace(".jpg", f"{self.suffix}.jpg")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert to tensors and normalize
        image = transforms.ToTensor()(image)  # [1, H, W]
        mask = transforms.ToTensor()(mask)    # [1, H, W]

        # Ensure binary mask
        mask = (mask > 0.5).float()

        return image, mask, image_name
