import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nii_to_png_slices():
    """
    For each .nii.gz in IMG_DIR, extracts all axial slices and
    saves them as grayscale PNGs under OUT_ROOT/images.
    For each matching mask in MSK_DIR, creates four binary masks
    (labels 1,2,3,4) and saves under OUT_ROOT/masks_wall, masks_cavity,
    masks_myoma, masks_cyst.
    """
    # ** USER-SPECIFIED PATHS **
    IMG_DIR = r"D:\qy44lyfe\LLM segmentation\Data sets\Uterine myoma MRI\Dataset003_UMD\imagesTr"
    MSK_DIR = r"D:\qy44lyfe\LLM segmentation\Data sets\Uterine myoma MRI\Dataset003_UMD\labelsTr"
    OUT_ROOT = r"D:\qy44lyfe\LLM segmentation\Data sets\utrine myoma converted"

    # output subfolders
    out_img    = os.path.join(OUT_ROOT, "images")
    out_wall   = os.path.join(OUT_ROOT, "masks_wall")
    out_cavity = os.path.join(OUT_ROOT, "masks_cavity")
    out_myoma  = os.path.join(OUT_ROOT, "masks_myoma")
    out_cyst   = os.path.join(OUT_ROOT, "masks_cyst")
    for d in (out_img, out_wall, out_cavity, out_myoma, out_cyst):
        mkdir(d)

    # find all training cases
    nifti_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.nii.gz")))

    for img_path in tqdm(nifti_paths, desc="Cases"):
        base = os.path.basename(img_path).replace("_t2_0000.nii.gz", "")
        # load image + mask
        img_nii = nib.load(img_path)
        msk_path = os.path.join(MSK_DIR, base + "_t2.nii.gz")
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask not found for {base}")
        msk_nii = nib.load(msk_path)

        img_vol = img_nii.get_fdata().astype(np.float32)
        msk_vol = msk_nii.get_fdata().astype(np.int16)

        # scale image intensities to [0,255]
        i_min, i_max = img_vol.min(), img_vol.max()
        img_scaled = ((img_vol - i_min) / (i_max - i_min) * 255).astype(np.uint8)

        Z = img_scaled.shape[2]
        for z in range(Z):
            slice_img = img_scaled[:, :, z]
            slice_msk = msk_vol[:, :, z]

            fn = f"{base}_slice{z:03d}.png"
            Image.fromarray(slice_img).save(os.path.join(out_img, fn))

            # four binary masks
            wall   = (slice_msk == 1).astype(np.uint8) * 255
            cavity = (slice_msk == 2).astype(np.uint8) * 255
            myoma  = (slice_msk == 3).astype(np.uint8) * 255
            cyst   = (slice_msk == 4).astype(np.uint8) * 255

            Image.fromarray(wall).save(os.path.join(out_wall, fn))
            Image.fromarray(cavity).save(os.path.join(out_cavity, fn))
            Image.fromarray(myoma).save(os.path.join(out_myoma, fn))
            Image.fromarray(cyst).save(os.path.join(out_cyst, fn))

if __name__ == "__main__":
    nii_to_png_slices()
