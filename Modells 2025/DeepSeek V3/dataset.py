import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, image_mask_suffix='_seg'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        self.transform = transform
        self.image_mask_suffix = image_mask_suffix

        # Get all PNG files in image directory
        self.image_files = [f for f in os.listdir(image_dir) if
                            f.endswith('.png') and not f.endswith(f'{image_mask_suffix}.png')]

        # Verify corresponding masks exist
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self._get_mask_name(img_file)
            if os.path.exists(os.path.join(self.mask_dir, mask_file)):
                self.valid_files.append(img_file)
            else:
                print(f"Warning: Mask not found for image {img_file}")

    def _get_mask_name(self, image_name):
        name, ext = os.path.splitext(image_name)
        return f"{name}{self.image_mask_suffix}{ext}"

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_name = self.valid_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self._get_mask_name(img_name))

        # Load image and mask
        image = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale numpy array
        mask = np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale numpy array

        # Binarize mask (assuming it's already binary, but just in case)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors and normalize image
        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor, img_name


def get_transforms(size=(256, 256)):
    return A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])