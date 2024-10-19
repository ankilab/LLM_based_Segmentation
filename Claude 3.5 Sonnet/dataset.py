import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_folder, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_folder
        self.transform = transform
        #self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_m.jpg')]

        #self.images.sort(key=lambda x: int(x.split('.')[0]))
        self.images.sort()  # Sorts lexicographically (alphabetically) (for brain dataset) ##########################

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        #mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_seg.png'))
        #mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '.png'))
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_m.jpg'))

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])