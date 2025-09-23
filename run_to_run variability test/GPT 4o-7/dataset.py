# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, mask_suffix='_m', transform_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.mask_suffix = mask_suffix
        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(self.image_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name.replace('.jpg', f'{self.mask_suffix}.jpg')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()  # ensure binary

        return image, mask, image_name
