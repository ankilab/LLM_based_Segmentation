import os
from PIL import Image
import numpy as np
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
            base, _ = os.path.splitext(fname)
            mask_name = f"{base}.png"
            img_path = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.isfile(mask_path):
                self.samples.append((img_path, mask_path, base))

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
        #image = Image.open(image_path).convert('L')
        # in case of RGB image data:
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        # Convert to NumPy arrays
        #image_np = np.array(image, dtype=np.float32) / 255.0   # shape (H, W), [0,1]
        # in case of RGB:
        image_np = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0     # shape (H, W), [0,1]
        mask_np = (mask_np > 0.5).astype(np.float32)           # binarize

        # To torch tensors, add channel dim
        #image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
        # in case of RGB:
        image_tensor = torch.from_numpy(image_np)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)    # [1, H, W]

        return image_tensor, mask_tensor, base_name
