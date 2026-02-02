import nibabel as nib
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import os
import cv2

#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\BAGLS_1077.nii.gz"  # Path to the .nii.gz file
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\Swallowing_0591.nii.gz"
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\BrainMeningioma_0260.nii.gz"
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\ISIC_4607.nii.gz" # ISIC_0028913.png —> ISIC_4607_0000.nii.gz
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\ISIC_0000.nii.gz" # ISIC_0024306.png —> ISIC_0000_0000.nii.gz
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\DRIVE_0031.nii.gz" # 11_g_HRF.jpg —> DRIVE_0031_0000.nii.gz
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\DRIVE_0011.nii.gz" # 04_h_HRF.jpg —> DRIVE_0011_0000.nii.gz
nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\MYOMA_0893.nii.gz" # case_0039_UMD_221129_055_slice013.png —> MYOMA_0893_0000.nii.gz
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\MYOMA_1611.nii.gz" # case_0068_UMD_221129_092_slice008.png —> MYOMA_1611_0000.nii.gz

save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\predictions\\"

#output_file = os.path.join(save_path, "nnUnet_BAGLS_1077.png")
#output_file = os.path.join(save_path, "nnUnet_Brain_0275.png")
#output_file = os.path.join(save_path, "nnUnet_Swallowing_0153.png")
#output_file = os.path.join(save_path, "nnUnet_ISIC_0024306.png")
#output_file = os.path.join(save_path, "nnUnet_11_g_HRF.png")
output_file = os.path.join(save_path, "nnUnet_case_0039_UMD_221129_055_slice013.png")

if not os.path.exists(save_path):
    os.makedirs(save_path)

nii_image = nib.load(nii_file_path)
nii_data = nii_image.get_fdata()
orig_h, orig_w = nii_data.shape

# Check 2D and normalize the data to 0-255 for PNG format
if nii_data.ndim == 2:  # Confirming it's 2D
    # Normalize the data to the range 0-255 and convert to uint8
    nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255
    nii_data = nii_data.astype(np.uint8)

    # Convert to PIL Image for resizing, rotating, and flipping
    img = Image.fromarray(nii_data)

    # Resize to 256x256
    #img = img.resize((256, 256), Image.ANTIALIAS)
    #img = img.resize((orig_h, orig_w), resample=Image.NEAREST)
    #img = img.resize((256, 256), resample=Image.Resampling.LANCZOS)

    # Resize to original dims with OpenCV (Lanczos for high quality)
    arr = np.array(img)
    resized = cv2.resize(arr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    img = Image.fromarray(resized)

    # # Boost sharpness (2.0 = twice as sharp)
    # sharpener = ImageEnhance.Sharpness(img)
    # img = sharpener.enhance(2.0)
    # # Boost contrast (1.5 = 50% more contrast)
    # contra = ImageEnhance.Contrast(img)
    # img = contra.enhance(1.5)

    # Rotate 90 degrees to the right and flip horizontally
    img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

    # Save as PNG
    img.save(output_file)
    print(f"Saved image as {output_file}")
else:
    print("Error: The loaded .nii.gz file is not 2D.")
