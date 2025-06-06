import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class GrayscaleDataset(Dataset):
    """
    Custom Dataset for loading grayscale images and their binary masks.
    Assumes:
      - `image_dir` contains only grayscale PNG files (e.g. "1.png", "2.png", …).
      - `mask_dir` contains corresponding binary masks named "<basename>_seg.png"
        (e.g. for "1.png" → mask "1_seg.png").
      - Both directories may contain other non-PNG files, which are ignored.
    """

    def __init__(self, image_dir, mask_dir, image_size=256):
        """
        Args:
            image_dir (str): Path to the folder containing input images.
            mask_dir (str): Path to the folder containing binary masks.
            image_size (int): Square size (width = height) to which images and masks are resized.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # Gather all valid image filenames (PNG only).
        all_files = sorted(os.listdir(self.image_dir))
        self.samples = []
        for fname in all_files:
            if not fname.lower().endswith('.png'):
                continue
            # We expect masks named "<basename>_seg.png"
            base, _ = os.path.splitext(fname)
            mask_name = f"{base}_seg.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            image_path = os.path.join(self.image_dir, fname)
            if os.path.isfile(mask_path):
                self.samples.append((image_path, mask_path, base))
            # else: skip images with no corresponding mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image_tensor (torch.FloatTensor): shape [1, H, W], values in [0,1]
            mask_tensor  (torch.FloatTensor): shape [1, H, W], values 0.0 or 1.0
            base_name    (str): filename without extension (e.g. "1", "2", …)
        """
        image_path, mask_path, base_name = self.samples[idx]

        # Load image and mask in PIL
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Resize
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
             .float()
             .view(self.image_size, self.image_size) / 255.0)
        ).unsqueeze(0)  # shape [1, H, W]

        mask_tensor = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
             .float()
             .view(self.image_size, self.image_size) / 255.0)
        ).unsqueeze(0)
        # Binarize mask
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor, base_name
