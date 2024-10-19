import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class BinarySegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_folder):
        self.image_dir = image_dir
        self.mask_dir = mask_folder
        #self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_m.jpg')]
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)

        #mask_path = os.path.join(self.mask_dir, image_name.replace('.png', '_seg.png'))
        #mask_path = os.path.join(self.mask_dir, image_name.replace('.png', '.png'))
        mask_path = os.path.join(self.mask_dir, image_name.replace('.jpg', '_m.jpg'))

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask