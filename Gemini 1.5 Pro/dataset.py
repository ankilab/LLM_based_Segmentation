import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_folder, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_folder
        #self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and not "_seg" in f], key=lambda x: int(x.split('.')[0])) #only png, ignore other files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and not "_m" in f])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        #mask_name = img_name.split('.')[0] + "_seg.png" #mask file naming
        #mask_name = img_name
        mask_name = img_name.split('.')[0] + "_m.jpg"

        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("L") # Ensure grayscale
        mask = Image.open(mask_path).convert("L")  # Ensure grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure binary mask (0 or 1)
        mask = (mask > 0).float()

        return image, mask