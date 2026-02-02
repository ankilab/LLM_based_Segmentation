# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, same_folder=False, suffix="_m", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        self.same_folder = same_folder
        self.suffix = suffix
        self.transform = transform

        # Only accept image files
        valid_ext = [".png", ".jpg", ".jpeg"]
        self.image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_ext and suffix not in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("L")

        # Build mask name
        base_name, ext = os.path.splitext(img_name)
        mask_name = f"{base_name}{self.suffix}{ext}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load mask
        mask = Image.open(mask_path).convert("L")

        # Resize and ToTensor
        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        image = to_tensor(resize(image))
        mask = to_tensor(resize(mask))

        # Binarize mask
        mask = (mask > 0.5).float()

        return image, mask, img_name
