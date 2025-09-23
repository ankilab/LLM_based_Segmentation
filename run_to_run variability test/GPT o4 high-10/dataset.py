# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GrayscaleSegmentationDataset(Dataset):
    """
    A Dataset for loading grayscale images and corresponding binary masks.
    Assumes images are .png files; masks have same basename plus optional suffix.
    """
    def __init__(self, image_dir, mask_dir=None, suffix='_m', transform=None):
        """
        Args:
            image_dir (str): path to folder with input images (.png)
            mask_dir (str): path to folder with masks (.png). If None, uses image_dir + suffix convention.
            suffix (str): suffix appended before .png in mask filenames
            transform (callable): transforms to apply to both image and mask
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.suffix = suffix
        self.transform = transform

        # List all pngs in image_dir
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base, _ = os.path.splitext(img_name)

        img_path = os.path.join(self.image_dir, img_name)
        if self.mask_dir == self.image_dir:
            mask_name = f"{base}{self.suffix}.jpg"
        else:
            mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('L')      # grayscale
        mask = Image.open(mask_path).convert('L')    # grayscale mask

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask, img_name

class PairedTransform:
    """Top-level class so DataLoader can pickle it."""
    def __init__(self, resize=(256,256)):
        self.tf_img  = T.Compose([ T.Resize(resize), T.ToTensor() ])
        self.tf_mask = T.Compose([ T.Resize(resize), T.ToTensor() ])

    def __call__(self, img, mask):
        return self.tf_img(img), self.tf_mask(mask)
