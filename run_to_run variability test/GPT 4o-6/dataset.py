# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, mask_suffix='_m', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir if mask_dir is not None else image_dir
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg') and not f.endswith(mask_suffix + '.jpg')
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace('.jpg', f'{self.mask_suffix}.jpg')

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Convert to tensor and normalize to [0,1]
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        image = transform(image)
        mask = transform(mask)
        mask = (mask > 0.5).float()

        return image, mask, image_name
