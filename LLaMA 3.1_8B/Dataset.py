python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '*.png')) # Update for other image formats if needed
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')
        mask = Image.open(self.mask_paths[index]).convert('L')

        image = np.array(image) / 255.0 # Normalize to [0, 1]
        mask = (np.array(mask) > 127).astype(np.float32) # Binarize mask

        if self.transform is not None:
            image, mask = self.transform(image), self.transform(mask)

        # Add channel dimension
        return torch.FloatTensor(image).unsqueeze(0), torch.FloatTensor(mask).unsqueeze(0)