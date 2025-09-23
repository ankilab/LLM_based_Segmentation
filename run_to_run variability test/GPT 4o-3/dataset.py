# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix="_m", image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.image_filenames = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') and not f.endswith(f'{suffix}.jpg')]
        self.suffix = suffix
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', f'{self.suffix}.jpg')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask, img_name
