# dataset.py
import os
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset

class UMDDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=(64,64,32), train=True):
        """
        image_paths: list of .nii.gz T2 volumes
        mask_paths : list of .nii.gz ground‐truth volumes
        patch_size : (H, W, D) target shape for every sample
        train      : whether to do any random augment (not shown here)
        """
        self.images     = image_paths
        self.masks      = mask_paths
        self.patch_size = patch_size  # (H, W, D)
        self.train      = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- 1) load raw data ---
        img_np = nib.load(self.images[idx]).get_fdata(dtype=np.float32)
        msk_np = nib.load(self.masks[idx]).get_fdata().astype(np.uint8)

        # --- 2) normalize intensity to [0,1] ---
        img_np = (img_np - img_np.min()) / (img_np.ptp() + 1e-8)

        # --- 3) add channel dim: [H,W,D] -> [1,H,W,D] ---
        img = torch.from_numpy(img_np).unsqueeze(0)   # float32
        msk = torch.from_numpy(msk_np).unsqueeze(0)   # uint8

        # --- 4) resample to patch_size ---
        #   note: interpolate expects [N,C,D,H,W] for trilinear, so add batch dim
        img = F.interpolate(img.unsqueeze(0),
                            size=self.patch_size,
                            mode='trilinear',
                            align_corners=False
                           ).squeeze(0)
        # masks: use nearest to preserve discrete labels
        msk = F.interpolate(msk.unsqueeze(0).float(),
                            size=self.patch_size,
                            mode='nearest'
                           ).squeeze(0).byte()

        # --- 5) optionally do augmentations here if train=True ---

        return img, msk
