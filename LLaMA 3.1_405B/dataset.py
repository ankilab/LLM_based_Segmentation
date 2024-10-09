import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BinarySegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.image_dir, image_name.replace('.png', '_seg.png'))

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask