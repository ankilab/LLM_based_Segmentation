import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".png") and not f.endswith("_seg.png")]

        # Define the resize transform to ensure uniformity across all images
        self.transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize both images and masks to target_size
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = image_name.replace(".png", "_seg.png")

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.image_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply the same resize transformation to both image and mask
        image = self.transform(image)
        mask = self.transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = torch.round(mask)

        return image, mask
