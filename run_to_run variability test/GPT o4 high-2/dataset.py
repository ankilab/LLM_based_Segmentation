# unet_segmentation/dataset.py

import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    """
    Dataset for loading grayscale images and binary masks (PNG only).
    Assumes images/ and masks/ in separate dirs with exact same filenames,
    or masks suffixed with _seg before the .png.
    """
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), mask_suffix='_m'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        # gather all png images
        self.images = sorted([p for p in glob(os.path.join(image_dir, '*.jpg'))])
        # map to masks
        self.masks = []
        for img_path in self.images:
            fname = os.path.basename(img_path)
            base, _ = os.path.splitext(fname)
            # look for base_seg.png first, else base.png in mask_dir
            seg_name = f"{base}{mask_suffix}.jpg"
            seg_path = os.path.join(mask_dir, seg_name)
            if os.path.isfile(seg_path):
                self.masks.append(seg_path)
            else:
                alt = os.path.join(mask_dir, fname)
                if os.path.isfile(alt):
                    self.masks.append(alt)
                else:
                    raise FileNotFoundError(f"Mask for {img_path} not found.")
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load
        img = Image.open(self.images[idx]).convert('L')
        m  = Image.open(self.masks[idx]).convert('L')  # binary mask
        # resize
        img = img.resize(self.img_size)
        m   = m.resize(self.img_size)
        # to tensor [0,1]
        img_t = TF.to_tensor(img)
        mask_t= TF.to_tensor(m)
        # binarize mask > 0.5
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t, os.path.basename(self.images[idx])
