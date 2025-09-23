# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix="_m", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.suffix = suffix
        self.transform = transform

        self.image_filenames = [
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") and not f.endswith(f"{self.suffix}.jpg")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = image_name.replace(".jpg", f"{self.suffix}.jpg")

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()  # Ensure binary

        return image, mask, image_name
