import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GrayscaleSegDataset(Dataset):
    """
    Dataset for loading grayscale images and binary masks.
    Expects image files (*.png) in image_dir and mask files (*.png) in mask_dir
    with matching base filenames, or in the same directory with a suffix.
    """
    def __init__(self, image_dir, mask_dir=None, suffix="_m", transform=None):
        """
        Args:
            image_dir (str): path to images folder.
            mask_dir (str, optional): path to masks folder. If None, masks are in image_dir.
            suffix (str): suffix appended to image filename (before extension) for mask.
            transform (callable, optional): transform to be applied to both image and mask (should handle both).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.suffix = suffix
        self.transform = transform

        # find all png images, ignore other files
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        # filter out those that look like masks if mask_dir == image_dir
        if self.mask_dir == self.image_dir:
            self.image_paths = [
                p for p in self.image_paths
                if not os.path.basename(p).endswith(f"{self.suffix}.jpg")
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = f"{base}{self.suffix}.jpg"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # load grayscale image
        image = Image.open(img_path).convert("L")
        # load mask and binarize (0 or 1)
        mask = Image.open(mask_path).convert("L").point(lambda p: 1 if p > 128 else 0, mode='1')

        if self.transform:
            image, mask = self.transform(image, mask)

        # ensure tensors: image 1xHxW float, mask 1xHxW long
        return image, mask.long()
