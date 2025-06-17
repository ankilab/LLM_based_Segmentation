import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

class GrayscaleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask_suffix='_seg'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = image_file.replace('.png', self.mask_suffix + '.png')
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Binarize the mask
        return image, mask