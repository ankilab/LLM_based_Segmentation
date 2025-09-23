# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, mask_suffix='_m', transform_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        self.mask_suffix = mask_suffix
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") and os.path.isfile(os.path.join(image_dir, f))]
        self.transform_img = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', f'{self.mask_suffix}.jpg')
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()

        return image, mask, img_name
