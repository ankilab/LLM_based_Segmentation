# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, suffix='_m', image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir else image_dir
        self.suffix = suffix
        self.image_size = image_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and suffix not in f])

        self.transform_img = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(image_size),
            T.ToTensor(),
        ])

        self.transform_mask = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', f'{self.suffix}.jpg')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return image, mask, img_name
