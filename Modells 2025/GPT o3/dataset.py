# unet_segmentation/dataset.py
import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    """
    Expects files like
        images_dir/
            0001.png
            0001_seg.png
            0002.png
            0002_seg.png
            ...
    and **ignores** anything that is not a .png or is a _seg.png without its image pair.
    """

    def __init__(
        self,
        images_dir: str | Path,
        resize: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        images_dir = Path(images_dir)
        self.resize = resize
        self.transform = transform
        self.samples: List[Tuple[Path, Path]] = []

        pngs = sorted(p for p in images_dir.glob("*.png") if not p.name.endswith("_seg.png"))
        for img_path in pngs:
            mask_path = img_path.with_name(img_path.stem + "_seg.png")
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess(self, pil_img: Image.Image, is_mask: bool = False) -> torch.Tensor:
        # Ensure grayscale, resize, to tensor, normalise (images 0-1; masks 0/1)
        pil_img = pil_img.convert("L").resize(self.resize, Image.NEAREST if is_mask else Image.BILINEAR)
        tensor = TF.to_tensor(pil_img)          # shape [1,H,W] in (0,1)
        if is_mask:
            tensor = (tensor > 0.5).float()     # strictly binary 0/1
        return tensor

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image = self._preprocess(Image.open(img_path), is_mask=False)
        mask  = self._preprocess(Image.open(mask_path), is_mask=True)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, img_path.name  # return id for later visualisation
