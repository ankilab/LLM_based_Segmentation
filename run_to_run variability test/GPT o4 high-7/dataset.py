import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Loads grayscale images and binary masks from two directories (or same directory with suffix).
    """
    def __init__(self, image_dir, mask_dir=None, mask_suffix='_m', transforms=None):
        """
        image_dir: path to images
        mask_dir: path to masks; if None, masks live alongside images
        mask_suffix: suffix before file extension for mask files, e.g. "_seg"
        transforms: torchvision transforms to apply to both image and mask
        """
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.mask_dir = mask_dir or image_dir
        self.mask_suffix = mask_suffix
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        name, ext = os.path.splitext(fname)
        mask_name = f"{name}{self.mask_suffix}{ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load as grayscale
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')  # binary mask as 0/255

        if self.transforms:
            image, mask = self.transforms(image, mask)

        # image: [1,H,W], mask: [1,H,W] with values 0 or 1
        return image, mask, fname


class ToTensorNormalize:
    """
    Composite transform to resize, to-tensor, and normalize image & mask.
    """
    def __init__(self, size):
        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.mask_transform = T.Compose([
            T.Resize(size),
            T.ToTensor()  # will be float [0,1]
        ])

    def __call__(self, img, mask):
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return img, mask
