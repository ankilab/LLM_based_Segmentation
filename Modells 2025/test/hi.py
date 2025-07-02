import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Hard-code your dataset path here:
ROOT = r"D:\qy44lyfe\LLM segmentation\Data sets\Uterine myoma MRI\Dataset003_UMD"
IMG_DIR   = os.path.join(ROOT, "imagesTr")
MASK_DIR  = os.path.join(ROOT, "labelsTr")

# Pick one case by filename (must match naming convention):
case_id = "case_0000_UMD_221129_001_t2_0000.nii.gz"
img_path  = os.path.join(IMG_DIR,  case_id)
mask_file = case_id.replace("_t2_0000.nii.gz", "_t2.nii.gz")
mask_path = os.path.join(MASK_DIR, mask_file)

# Load volumes
img_nii  = nib.load(img_path)
msk_nii  = nib.load(mask_path)
img      = img_nii.get_fdata()
mask     = msk_nii.get_fdata()

# Choose the middle axial slice
z = img.shape[2] // 2

# Plot
plt.figure(figsize=(8,4))

# T2 image
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img[:, :, z].T, cmap="gray", origin="lower")
ax1.set_title("T2 image (slice {})".format(z))
ax1.axis("off")

# Mask
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(mask[:, :, z].T, cmap="gray", origin="lower")
ax2.set_title("GT mask (slice {})".format(z))
ax2.axis("off")

plt.tight_layout()
plt.show()
