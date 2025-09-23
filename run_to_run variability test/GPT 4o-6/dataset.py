# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

class GraySegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix="_m", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.suffix = suffix
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") and suffix not in f]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", f"{self.suffix}.jpg")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("L")  # Grayscale
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize to [0,1], then binarize mask
        image = image / 255.0
        mask = (mask > 127).float()

        return image, mask, img_name
