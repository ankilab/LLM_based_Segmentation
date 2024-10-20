import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_folder, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_folder
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        #self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_m.jpg')]

        #self.masks = [f.replace('.png', '_seg.png') for f in self.images]
        self.masks = [f.replace('.png', '.png') for f in self.images]
        #self.masks = [f.replace('.jpg', '_m.jpg') for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask