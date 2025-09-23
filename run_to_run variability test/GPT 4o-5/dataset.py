# unet_segmentation/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, mask_suffix="_m", image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.mask_suffix = mask_suffix
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", f"{self.mask_suffix}.jpg")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Ensure binary mask

        return image, mask, img_name
