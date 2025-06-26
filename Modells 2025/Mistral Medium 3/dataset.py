import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, mask_suffix='_seg', transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str, optional): Directory with all the masks. If None, masks are assumed to be in the same directory as images with a suffix.
            mask_suffix (str): Suffix for mask filenames if masks are in the same directory as images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        #self.mask_dir = mask_dir
        self.mask_dir = mask_dir or image_dir
        self.mask_suffix = mask_suffix
        self.transform = transform

        # Get list of image files (only .png)
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.endswith('.png') and not f.endswith('_seg.png')]

        if mask_dir is None:
            # Masks are in the same directory with a suffix
            self.mask_files = [f.replace('.png', f'{mask_suffix}.png') for f in self.image_files]
        else:
            # Masks are in a separate directory with the same name
            #self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_seg.png')]
            self.mask_files = [f for f in self.image_files]

        # Ensure all masks exist
        missing_masks = []
        for img, mask in zip(self.image_files, self.mask_files):
            # if mask_dir is None:
            #     mask_path = os.path.join(image_dir, mask)
            # else:
            #     mask_path = os.path.join(mask_dir, mask)
            mask_path = os.path.join(self.mask_dir, mask)
            if not os.path.exists(mask_path):
                missing_masks.append(mask)

        if missing_masks:
            raise FileNotFoundError(f"Missing masks: {missing_masks}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        if self.mask_dir is None:
            mask_path = os.path.join(self.image_dir, self.mask_files[idx])
        else:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('L')  # Ensure grayscale
        mask = Image.open(mask_path).convert('L')   # Ensure binary mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask