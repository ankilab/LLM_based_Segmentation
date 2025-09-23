# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    Dataset for grayscale images with binary masks.
    Finds all .png/.jpg images in images_dir and their masks in masks_dir (or same dir)
    by inserting `mask_suffix` before the file extension.
    """

    def __init__(self,
                 images_dir,
                 masks_dir=None,
                 mask_suffix='_m',
                 img_size=(256, 256),
                 img_exts=('.png', '.jpg', '.jpeg')):
        """
        Args:
            images_dir (str): folder with input images.
            masks_dir  (str): folder with masks (if different); if None, same as images_dir.
            mask_suffix (str): suffix to insert before the extension of the image filename.
            img_size   (tuple): (H, W) to resize both image and mask.
            img_exts   (tuple): allowed image file extensions.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.img_exts = img_exts

        # collect all image filenames with allowed extensions
        self.image_files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith(self.img_exts)
        )

        # transforms
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = T.Compose([
            T.Resize(self.img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),  # yields [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # insert mask_suffix before extension
        base, ext = os.path.splitext(img_name)
        mask_name = base + self.mask_suffix + ext
        mask_path = os.path.join(self.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # load
        image = Image.open(img_path)
        mask  = Image.open(mask_path)

        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()  # ensure binary mask

        return image, mask, img_name
