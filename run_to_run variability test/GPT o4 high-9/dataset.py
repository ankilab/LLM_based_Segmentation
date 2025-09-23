# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GrayscaleSegmentationDataset(Dataset):
    """
    Loads grayscale images and their binary masks from a single folder or two separate folders.
    Only .png files are considered; other files are ignored.
    """
    def __init__(self, images_dir, masks_dir=None, suffix="_m.jpg", transform=None):
        """
        If masks_dir is None, assumes masks are in the same folder as images_dir,
        with the same base name plus `suffix`.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.suffix = suffix
        self.transform = transform

        # gather image files
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg") and not f.lower().endswith(suffix)]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name if self.masks_dir != self.images_dir else img_name.replace(".jpg", f"{self.suffix}")
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("L")
        mask  = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform((image, mask))

        # convert to tensor
        image = transforms.ToTensor()(image)
        mask  = transforms.ToTensor()(mask)
        # ensure binary mask (0 or 1)
        mask  = (mask > 0.5).float()

        return image, mask, img_name
