import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert('L')  # Grayscale image
        mask = Image.open(mask_path).convert('L')  # Binary mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
