import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Ensure we only load PNG files
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.png') and '_seg' not in img]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.png', '_seg.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.image_dir, mask_name)

        # Load image and mask, and ensure they are PNG files
        try:
            image = Image.open(img_path).convert("L")  # Ensure grayscale conversion
            mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale
        except Image.UnidentifiedImageError:
            raise ValueError(f"File {img_path} or {mask_path} is not a valid PNG image.")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
