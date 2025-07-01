import os
import torch
import nibabel as nib
import numpy as np
import math
from torch.utils.data import Dataset

class UMDDataset(Dataset):
    """
    3D UMD Dataset with on‐the‐fly patch sampling.
    Half the time samples a patch containing myoma, half random background.
    Pads to multiple of 16 before patching.
    """
    def __init__(self, image_paths, mask_paths,
                 patch_size=(64,64,32), train=True):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.patch_size  = patch_size
        self.train       = train

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _pad_to_multiple(vol, m=16):
        D,H,W = vol.shape
        tD,tH,tW = [math.ceil(x/m)*m for x in (D,H,W)]
        pd,ph,pw = tD-D, tH-H, tW-W
        pad = ((pd//2,pd-pd//2),(ph//2,ph-ph//2),(pw//2,pw-pw//2))
        return np.pad(vol, pad, mode="constant", constant_values=0)

    def __getitem__(self, idx):
        # load and normalize
        img = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        img = (img - img.mean())/(img.std()+1e-8)
        mask = nib.load(self.mask_paths[idx]).get_fdata().astype(np.int16)
        mask = (mask == 3).astype(np.float32)

        # pad to multiple of 16
        img  = self._pad_to_multiple(img)
        mask = self._pad_to_multiple(mask)

        # to torch
        img_t  = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.train:
            # patch sampling
            C,D,H,W = img_t.shape
            pd,ph,pw = self.patch_size

            if torch.rand(1) < 0.5 and mask_t.sum()>0:
                # sample foreground-centered patch
                fg = torch.nonzero(mask_t[0], as_tuple=False)
                z,y,x = fg[torch.randint(len(fg),(1,))][0]
                z0 = max(0, z-pd//2)
                y0 = max(0, y-ph//2)
                x0 = max(0, x-pw//2)
            else:
                # random patch
                z0 = torch.randint(0, D-pd+1,(1,)).item()
                y0 = torch.randint(0, H-ph+1,(1,)).item()
                x0 = torch.randint(0, W-pw+1,(1,)).item()

            img_t  = img_t[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]
            mask_t = mask_t[:,z0:z0+pd, y0:y0+ph, x0:x0+pw]

        return img_t.clone(), mask_t.clone()
