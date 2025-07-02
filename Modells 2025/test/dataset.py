# dataset.py
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class UMDDataset(Dataset):
    """
    Patch‐based dataset for training and validation.
    When train=True: random crop to patch_size.
    When train=False: center crop to patch_size.
    """
    def __init__(self, images, masks, patch_size=(64,64,32), train=True):
        self.images     = images
        self.masks      = masks
        self.patch_size = patch_size
        self.train      = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load as float32
        img = nib.load(self.images[idx]).get_fdata().astype(np.float32)
        msk = nib.load(self.masks[idx]).get_fdata().astype(np.float32)

        ph, pw, pd = self.patch_size
        dh, dw, dd = img.shape

        # pad if smaller
        pad_h = max(0, ph - dh)
        pad_w = max(0, pw - dw)
        pad_d = max(0, pd - dd)
        if pad_h or pad_w or pad_d:
            pad_cfg = (
                (pad_h//2, pad_h - pad_h//2),
                (pad_w//2, pad_w - pad_w//2),
                (pad_d//2, pad_d - pad_d//2),
            )
            img = np.pad(img, pad_cfg, mode="constant", constant_values=0)
            msk = np.pad(msk, pad_cfg, mode="constant", constant_values=0)
            dh, dw, dd = img.shape

        # add channel dim
        img = img[None]  # 1×H×W×D
        msk = msk[None]

        # crop
        if self.train:
            y0 = np.random.randint(0, dh - ph + 1)
            x0 = np.random.randint(0, dw - pw + 1)
            z0 = np.random.randint(0, dd - pd + 1)
        else:
            # center crop
            y0 = (dh - ph) // 2
            x0 = (dw - pw) // 2
            z0 = (dd - pd) // 2

        img = img[:, y0:y0+ph, x0:x0+pw, z0:z0+pd]
        msk = msk[:, y0:y0+ph, x0:x0+pw, z0:z0+pd]

        # to torch.Tensor
        return torch.from_numpy(img), torch.from_numpy(msk)
