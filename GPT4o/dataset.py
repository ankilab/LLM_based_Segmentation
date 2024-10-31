import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = [f for f in os.listdir(image_dir) if f.endswith(".png") and not f.endswith("_seg.png")]
        #self.image_paths = [f for f in os.listdir(image_dir) if f.endswith(".jpg") and not f.endswith("_m.jpg")]
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # bolus dataset
        #mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_seg.png')) # BAGLS dataset
        #mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_m.jpg'))  # Brain tumor dataset

        image = Image.open(img_path).convert('L')  # Grayscale
        mask = Image.open(mask_path).convert('L')  # Binary mask

        # Resize images and masks to the specified size
        resize_transform = T.Resize(self.image_size)
        image = resize_transform(image)
        mask = resize_transform(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.where(mask > 0.5, 1.0, 0.0)  # Binarize mask

        return image, mask
