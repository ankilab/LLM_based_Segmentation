import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GrayscaleSegmentationDataset(Dataset):
    """
    Loads grayscale images and their binary masks from a single folder or separate
    folders. Masks must end in '_seg.png' (or same name in a mask_folder).
    Ignores any non-png files.
    """
    def __init__(self, image_folder, mask_folder=None, mask_suffix='_m.jpg', transform=None):
        """
        image_folder: path to images
        mask_folder: optional separate path to masks; if None, masks in image_folder
        mask_suffix: if masks are named like '001_seg.png'
        transform: torchvision.transforms to apply to both image & mask
        """
        self.image_folder  = image_folder
        self.mask_folder   = mask_folder or image_folder
        self.mask_suffix   = mask_suffix
        self.transform     = transform

        # collect all png images
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith('.jpg') and not f.lower().endswith(mask_suffix)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base      = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = base + self.mask_suffix
        mask_path = os.path.join(self.mask_folder, mask_name)

        # open
        image = Image.open(img_path).convert('L')
        mask  = Image.open(mask_path).convert('L')  # still one channel

        if self.transform:
            # apply same transforms to image and mask
            image, mask = self.transform(image, mask)

        return image, mask, os.path.basename(img_path)
