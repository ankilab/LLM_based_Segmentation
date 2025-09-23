# unet_segmentation/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    A Dataset loading paired grayscale images and binary masks (PNG only).
    If masks_dir is None, looks for masks in images_dir using mask_suffix.
    """
    def __init__(self,
                 images_dir: str,
                 masks_dir: str = None,
                 mask_suffix: str = "_m",
                 img_size=(256,256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        # collect image files
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        # ensure corresponding masks exist
        self._pairs = []
        for img_name in self.images:
            name, _ = os.path.splitext(img_name)
            mask_name = name + mask_suffix + ".jpg"
            if os.path.exists(os.path.join(self.masks_dir, mask_name)):
                self._pairs.append((img_name, mask_name))
        # transforms: resize, to tensor
        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),             # will be 1×H×W in [0,1]
        ])
        self.mask_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),             # 1×H×W in [0,1]
        ])

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self._pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        # load
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        img = self.img_transform(img)                       # [1,H,W]
        mask = self.mask_transform(mask)
        # binarize
        mask = (mask > 0.5).float()
        return img, mask, img_name
