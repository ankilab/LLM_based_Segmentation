# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class SegmentationDataset(Dataset):
    """
    Dataset for loading grayscale images and their binary masks.
    Masks are PNG files with the same basename plus an optional suffix.
    """

    def __init__(self,
                 images_dir: str,
                 masks_dir: str = None,
                 mask_suffix: str = "_m.jpg",
                 image_ext: str = ".jpg",
                 transform=None,
                 img_size=(256, 256)):
        """
        images_dir: folder containing input PNGs
        masks_dir: if None, masks are in images_dir with suffix
        mask_suffix: suffix to append to basename to find mask
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.mask_suffix = mask_suffix
        self.image_ext = image_ext
        self.transform = transform
        self.img_size = img_size

        # list all image files
        all_files = os.listdir(images_dir)
        # keep only .png files not ending with suffix
        self.image_files = sorted(
            [f for f in all_files
             if f.endswith(image_ext) and not f.endswith(mask_suffix)]
        )

        # define default transforms if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(self.img_size),
                T.ToTensor(),
                # image is grayscale -> single channel [0,1]
            ])
        # mask transform (resize + to tensor only)
        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base, _ = os.path.splitext(img_name)
        mask_name = base + self.mask_suffix
        mask_path = os.path.join(self.masks_dir, mask_name)

        # load image and mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)  # [1,H,W]
        mask = self.mask_transform(mask)  # [1,H,W]
        # binarize mask
        mask = (mask > 0.5).float()

        return image, mask, img_name
