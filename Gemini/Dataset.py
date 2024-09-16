from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension for binary mask

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask