# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    """
    A Dataset for paired grayscale images and binary masks,
    with flexible mask-name matching (exact, suffix, or partial).
    Resizes all samples to a fixed size so batches collate cleanly.
    """
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}

    def __init__(self,
                 images_dir: str,
                 masks_dir: str = None,
                 mask_suffix: str = '_m',
                 resize: tuple = (256, 256),
                 transform=None):
        """
        images_dir: folder with inputs
        masks_dir: folder with masks (if None, assumed same as images_dir)
        mask_suffix: appended to basename for mask names in single-dir mode
        resize:  (H, W) for PIL resizing both images & masks
        transform: optional callable(img_tensor, mask_tensor) -> (img, mask)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir or images_dir
        self.mask_suffix = mask_suffix
        self.resize = resize
        self.transform = transform

        # collect image filenames
        self.img_names = [
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTS
               and not (masks_dir is None
                        and os.path.splitext(f)[0].endswith(mask_suffix))
        ]
        if not self.img_names:
            raise RuntimeError(f"No images found in {images_dir}")
        self.img_names.sort()

        # pre-scan mask directory once
        self.mask_names = [
            f for f in os.listdir(self.masks_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTS
        ]
        if not self.mask_names:
            raise RuntimeError(f"No masks found in {self.masks_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base, ext = os.path.splitext(img_name)

        # find matching mask
        if self.masks_dir != self.images_dir:
            # 1) exact basename match
            candidates = [m for m in self.mask_names
                          if os.path.splitext(m)[0] == base]
            # 2) basename + suffix
            if not candidates:
                candidates = [m for m in self.mask_names
                              if os.path.splitext(m)[0] == base + self.mask_suffix]
            # 3) any mask containing both base and suffix
            if not candidates:
                candidates = [m for m in self.mask_names
                              if base in m and self.mask_suffix in m]
            # 4) any mask containing base at all
            if not candidates:
                candidates = [m for m in self.mask_names
                              if base in os.path.splitext(m)[0]]
            if not candidates:
                raise FileNotFoundError(f"No mask for '{img_name}' in {self.masks_dir}")
            # pick first
            mask_name = candidates[0]

        else:
            # same-dir: base + suffix + ext
            mask_name = base + self.mask_suffix + ext
            if mask_name not in self.mask_names:
                raise FileNotFoundError(f"No mask '{mask_name}' for '{img_name}'")

        mask_path = os.path.join(self.masks_dir, mask_name)

        # load & resize
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        img = img.resize(self.resize, Image.BILINEAR)
        mask = mask.resize(self.resize, Image.NEAREST)

        # to tensor
        img = TF.to_tensor(img)  # shape [1,H,W]
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask, img_name
