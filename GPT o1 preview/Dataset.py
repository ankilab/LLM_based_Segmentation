# dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the images directory.
            masks_dir (str): Path to the masks directory.
            transform (callable, optional): Transform to be applied to both images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.mask_names = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        image = Image.open(img_path).convert("L")  # Grayscale image
        mask = Image.open(mask_path).convert("L")  # Grayscale mask (binary)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Default to tensor conversion if no transform is provided
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask
