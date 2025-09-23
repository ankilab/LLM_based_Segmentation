# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading paired grayscale images and binary masks.
    Assumes:
      - Images are .png in img_dir
      - Masks are .png in mask_dir, with identical filenames or with a suffix
        (specified via mask_suffix)
    """
    def __init__(self, img_dir, mask_dir=None, mask_suffix='_m', transforms=None):
        """
        Args:
            img_dir (str): Path to images folder.
            mask_dir (str, optional): Path to masks folder. If None, masks are
                assumed to live in img_dir.
            mask_suffix (str): Suffix added to image basename to get mask name.
            transforms (callable, optional): transform(image, mask) -> (img, mask)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir or img_dir
        self.mask_suffix = mask_suffix
        self.transforms = transforms

        # list all png images, ignore other files
        self.img_names = [f for f in os.listdir(self.img_dir)
                          if f.lower().endswith('.jpg')]
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # build mask name
        base, ext = os.path.splitext(img_name)
        mask_name = f"{base}{self.mask_suffix}{ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # open images
        img = Image.open(img_path).convert('L')   # grayscale
        mask = Image.open(mask_path).convert('L') # binary

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask, img_name
