# unet_segmentation/dataset.py

import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading grayscale images and their binary masks.
    Expects PNG files; ignores any other files.
    """
    def __init__(self, image_dir, mask_dir=None, suffix='_m.jpg',
                 file_list=None, transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Directory with input images.
            mask_dir (str): Directory with masks. If None, assumes masks live in image_dir with suffix.
            suffix (str): Mask filename suffix if mask_dir is None.
            file_list (list of str): Basenames (without extension) to use; if None, auto-discovers.
            transform (callable): transforms for input image.
            mask_transform (callable): transforms for mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.suffix = suffix
        self.transform = transform or T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),             # gives [0,1]
        ])
        self.mask_transform = mask_transform or T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),             # gives [0,1]
        ])

        if file_list:
            self.basenames = file_list
        else:
            # discover png images ignoring those ending with suffix
            all_imgs = glob(os.path.join(image_dir, '*.jpg'))
            self.basenames = [os.path.splitext(os.path.basename(p))[0]
                              for p in all_imgs if not p.endswith(suffix)]
        self.basenames.sort()

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        name = self.basenames[idx]
        img_path = os.path.join(self.image_dir, name + '.jpg')
        mask_name = name + (self.suffix if self.mask_dir==self.image_dir else '.jpg')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('L')   # grayscale
        mask = Image.open(mask_path).convert('L')   # grayscale mask

        image = self.transform(image)
        mask = self.mask_transform(mask)
        # ensure binary: threshold at 0.5
        mask = (mask > 0.5).float()
        return image, mask, name
