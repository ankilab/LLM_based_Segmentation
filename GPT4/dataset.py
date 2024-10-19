# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

class CustomDataset(Dataset):
    def __init__(self, data_folder, mask_folder, transform=None):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        #self.image_filenames = [f for f in os.listdir(data_folder) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.image_filenames = [f for f in os.listdir(data_folder) if f.endswith('.jpg') and not f.endswith('_m.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, self.image_filenames[idx])
        #mask_path = os.path.join(self.mask_folder, self.image_filenames[idx].replace('.png', '_seg.png')) # BAGLS
        #mask_path = os.path.join(self.mask_folder, self.image_filenames[idx]) # bolus
        mask_path = os.path.join(self.mask_folder, self.image_filenames[idx].replace('.jpg', '_m.jpg')) # brain

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_transform():
    return Compose([
        Resize((256, 256)),
        ToTensor(),
    ])
