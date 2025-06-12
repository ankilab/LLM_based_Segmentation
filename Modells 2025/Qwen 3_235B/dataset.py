import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class GrayscaleBinaryMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, mask_suffix='', img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        # Transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_name = os.path.splitext(image_name)[0]
        mask_name = f"{base_name}{self.mask_suffix}.png"

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load images
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Apply transformations
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Threshold mask to binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask, image_name