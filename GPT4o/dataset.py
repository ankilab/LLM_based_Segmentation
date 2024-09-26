import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        # Filter only .png files and ignore others
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith('.png') and '_seg' not in f]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_filenames[idx])
        mask_path = image_path.replace(".png", "_seg.png")

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask to the same target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
