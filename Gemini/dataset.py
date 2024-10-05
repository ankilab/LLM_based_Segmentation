import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.mask_paths = [os.path.splitext(f)[0] + '_seg.png' for f in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        mask_path = os.path.join(self.data_dir, self.mask_paths[idx])

        image = Image.open(image_path).convert('L')  # Ensure grayscale
        mask = Image.open(mask_path).convert('L')  # Ensure binary mask

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Normalize image and add channel dimension
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

        return image, mask