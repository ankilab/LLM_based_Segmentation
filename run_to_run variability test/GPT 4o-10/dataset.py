# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix='_m', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.suffix = suffix
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith(f'{suffix}.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if self.mask_dir:
            mask_name = img_name.replace('.jpg', f'{self.suffix}.jpg')
            mask_path = os.path.join(self.mask_dir, mask_name)
        else:
            raise ValueError("mask_dir must be provided or be the same as image_dir")

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()  # binary mask

        return image, mask, img_name
