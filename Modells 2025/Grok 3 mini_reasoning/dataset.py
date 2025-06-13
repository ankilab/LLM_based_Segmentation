import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') and not f.endswith('_seg.png')]
        self.mask_files = [f.replace('.png', '_seg.png') for f in self.image_files if os.path.exists(os.path.join(root_dir, f.replace('.png', '_seg.png')))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        mask_name = os.path.join(self.root_dir, self.mask_files[idx])

        image = Image.open(img_name).convert('L')  # Convert to grayscale
        mask = Image.open(mask_name).convert('L')   # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # Apply the same transform to mask

        # Ensure mask is binary (threshold to 0 or 1)
        mask = (mask > 0.5).float()  # Assuming masks are 0-255, threshold at 0.5 after normalization

        return image, mask

# Example transform (can be imported and used in main.py)
def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),          # Convert to tensor (scales to 0-1)
    ])