# dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), transform=None):
        self.image_dir = image_dir  # Path to images
        self.mask_dir = mask_dir    # Path to masks
        self.transform = transform
        self.image_size = image_size
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.png') and not img.endswith('_seg.png')]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # mask_name = img_name.replace('.png', '_seg.png')
        mask_name = img_name

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Resize images and masks
        resize = transforms.Resize(self.image_size)
        image = resize(image)
        mask = resize(mask)

        # Apply any transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert images to tensors
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  # Binarize mask

        return image, mask, img_name
