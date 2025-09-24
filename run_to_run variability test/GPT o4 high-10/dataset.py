# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GrayscaleSegmentationDataset(Dataset):
    """
    Loads grayscale images and corresponding binary masks from a directory.
    Images and masks must be PNGs; masks share the same filename with optional suffix.
    """

    def __init__(self, images_dir, masks_dir=None, mask_suffix='_m', transform=None, mask_transform=None):
        """
        images_dir: path to folder with input PNGs
        masks_dir: path to folder with mask PNGs (if None, same as images_dir)
        mask_suffix: suffix before the .png in mask filenames, e.g. 'img.png'→'img_seg.png'
        transform: torchvision transforms for images
        mask_transform: torchvision transforms for masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mask_transform = mask_transform

        # list all PNG images in images_dir, ignore other files
        self.ids = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # derive mask filename
        base, ext = os.path.splitext(img_name)
        mask_name = f"{base}{self.mask_suffix}{ext}"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # load, convert to L
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')  # grayscale mask

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # ensure binary: values 0 or 1
            mask = (mask > 0.5).float()

        return image, mask, img_name
