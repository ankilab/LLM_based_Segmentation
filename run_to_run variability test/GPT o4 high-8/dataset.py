# dataset.py

import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GrayscaleSegmentationDataset(Dataset):
    """
    Args:
        images_dir (str): Path to folder with input .png images.
        masks_dir (str): Path to folder with corresponding mask .png files.
                         Masks must have same basename as image, or with suffix '_seg'.
        image_suffix (str): If masks have suffix, e.g. '_seg', set image_suffix='_seg'.
        transform (callable, optional): Optional transform to be applied
            on a sample (tuple of PIL Images).
    """
    def __init__(self, images_dir, masks_dir=None, image_suffix='_m', transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.image_suffix = image_suffix
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
        ])
        # collect all png image paths, ignore non-png
        self.image_paths = sorted(glob(os.path.join(self.images_dir, "*.jpg")))
        # filter out mask files if in same dir
        if self.images_dir == self.masks_dir:
            self.image_paths = [p for p in self.image_paths if not p.endswith(f"{self.image_suffix}.jpg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = base + self.image_suffix + ".jpg"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # load
        img = Image.open(img_path).convert('L')    # grayscale
        mask = Image.open(mask_path).convert('L')  # grayscale mask

        # apply same resize
        img = self.transform(img)
        mask = self.transform(mask)

        # to tensor & normalize
        img = transforms.functional.to_tensor(img)           # [0,1]
        mask = transforms.functional.to_tensor(mask)
        mask = (mask > 0.5).float()                         # binary 0/1

        return img, mask, os.path.basename(img_path)
