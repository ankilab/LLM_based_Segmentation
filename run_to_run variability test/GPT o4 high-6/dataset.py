# unet_segmentation/dataset.py

import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading grayscale images and their binary masks.
    You can now pass in the mask suffix (e.g. "_m") and mask extension (e.g. ".jpg").
    """
    def __init__(self,
                 image_dir,
                 mask_dir=None,
                 mask_suffix='_m',
                 mask_ext='.jpg',
                 file_list=None,
                 transform=None,
                 mask_transform=None):
        """
        Args:
            image_dir (str): Directory with input images.
            mask_dir (str): Directory with masks. If None, assumes masks live in image_dir.
            mask_suffix (str): suffix to append to the image basename to get the mask basename.
            mask_ext (str): extension (including dot) for mask files.
            file_list (list of str): Basenames (no extension) to use; if None, auto-discovers.
            transform, mask_transform: torchvision transforms for image/mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext

        self.transform = transform or T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
        ])
        self.mask_transform = mask_transform or T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
        ])

        if file_list:
            self.basenames = file_list
        else:
            # discover all image files ending in .png or .jpg, excluding masks
            all_imgs = []
            for ext in ['.png', '.jpg', '.jpeg']:
                all_imgs += glob(os.path.join(image_dir, f'*{ext}'))
            # remove any whose name already ends with mask_suffix + ext
            self.basenames = []
            for p in all_imgs:
                stem = os.path.splitext(os.path.basename(p))[0]
                if not stem.endswith(mask_suffix):
                    self.basenames.append(stem)
            self.basenames.sort()

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        name = self.basenames[idx]
        # image: try .png then .jpg
        for img_ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.image_dir, name + img_ext)
            if os.path.isfile(img_path):
                break
        else:
            raise FileNotFoundError(f"Image file not found for base {name}")

        mask_name = name + self.mask_suffix
        mask_path = os.path.join(self.mask_dir, mask_name + self.mask_ext)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert('L')
        mask  = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()
        return image, mask, name
