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
    Two ways to arrange your data:

    1. **Same folder** (original behaviour — now commented out below):
           images_dir/
               0001.png
               0001_seg.png
               0002.png
               0002_seg.png
               ...

    2. **Separate folders** (images + masks share the *basename* or the mask
       may have an extra ``_m`` suffix):
           images_dir/
               0001.png
               0002.png
           masks_dir/
               0001.png          # or 0001_m.png
               0002.png          # or 0002_m.png
    Anything that is not a ``.png`` or does not have a matching mask is ignored.
    """

    def __init__(
        self,
        images_dir: str | Path,
        resize: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        *,
        masks_dir: str | Path | None = None,
        mask_suffix: str = "",  # change to "" if masks share exact filename
    ) -> None:
        super().__init__()
        images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else images_dir
        self.mask_suffix = mask_suffix
        self.resize = resize
        self.transform = transform
        self.samples: List[Tuple[Path, Path]] = []

        # All candidate images (ignore existing *_seg.png masks etc.)
        # pngs = sorted(p for p in images_dir.glob("*.png") if not p.name.endswith(f"{mask_suffix}.png"))
        pngs = sorted(
            p for p in images_dir.glob("*.png")
            if not (mask_suffix and p.name.endswith(f"{mask_suffix}.png"))
        )

        for img_path in pngs:
            mask_path = self._find_mask(img_path)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image–mask pairs found in {images_dir}"
                + (f" with masks in {self.masks_dir}" if masks_dir else "")
            )

    # --------------------------------------------------------------------- #
    #                               Helpers                                 #
    # --------------------------------------------------------------------- #
    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """
        Locate the mask path corresponding to ``img_path``.
        Order of checks (stop at the first hit):

        1. masks_dir / (stem + mask_suffix + ".png")   e.g. 0001_seg.png
        2. masks_dir / (stem + "_m.png")               e.g. 0001_m.png
        3. masks_dir / (img.name)                      e.g. 0001.png

        Returns None if nothing exists.
        """
        stem = img_path.stem
        cands = [
            self.masks_dir / f"{stem}{self.mask_suffix}.png",
            self.masks_dir / f"{stem}_m.png",
            self.masks_dir / img_path.name,
        ]
        for p in cands:
            if p.exists():
                return p
        return None

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
