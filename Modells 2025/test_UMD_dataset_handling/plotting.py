import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT    = r"D:\qy44lyfe\LLM segmentation\Data sets\Uterine myoma MRI\Dataset003_UMD"
IMG_DIR = os.path.join(ROOT, "imagesTr")
MSK_DIR = os.path.join(ROOT, "labelsTr")

def load_nii(path):
    return nib.load(path)

# pick one case
case_id  = "case_0000_UMD_221129_001_t2_0000.nii.gz"
img_path  = os.path.join(IMG_DIR,  case_id)
mask_file = case_id.replace("_t2_0000.nii.gz", "_t2.nii.gz")
mask_path = os.path.join(MSK_DIR, mask_file)

# load
img_nii = load_nii(img_path)
msk_nii = load_nii(mask_path)

img = img_nii.get_fdata()             # float64 by default
msk = msk_nii.get_fdata()
msk = msk.astype(np.int16)            # now you have integer mask

# choose slice
z = img.shape[2] // 2

# plot
plt.figure(figsize=(12,4))

ax = plt.subplot(1,3,1)
ax.imshow(img[:, :, z].T, cmap="gray", origin="lower")
ax.set_title(f"T2 image (slice {z})")
ax.axis("off")

ax = plt.subplot(1,3,2)
ax.imshow(msk[:, :, z].T, cmap="gray", origin="lower")
ax.set_title(f"GT mask (slice {z})")
ax.axis("off")

# if you also have a predicted mask array `pred`, do:
# ax = plt.subplot(1,3,3)
# ax.imshow(pred[:, :, z].T, cmap="gray", origin="lower")
# ax.set_title(f"Pred mask (slice {z})")
# ax.axis("off")

plt.tight_layout()
plt.show()
