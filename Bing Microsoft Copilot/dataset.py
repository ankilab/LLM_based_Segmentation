import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

#
# class SegmentationDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
#         self.mask_files = [f.replace('.png', '_seg.png') for f in self.image_files]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         mask_path = os.path.join(self.image_dir, self.mask_files[idx])
#         image = Image.open(img_path).convert('L')
#         mask = Image.open(mask_path).convert('L')
#
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
#
#         return image, mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_folder, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_folder
        self.transform = transform
        self.target_size = target_size

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        #self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_m.jpg')]

        #self.mask_files = [f.replace('.png', '_seg.png') for f in self.image_files]
        self.mask_files = [f.replace('.png', '.png') for f in self.image_files]
        #self.mask_files = [f.replace('.jpg', '_m.jpg') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Resize images and masks
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
