import nibabel as nib
import numpy as np
from PIL import Image
import os

#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\BAGLS_1077.nii.gz"  # Path to the .nii.gz file
#nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\Swallowing_0591.nii.gz"
nii_file_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\BrainMeningioma_0260.nii.gz"

save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline"
#output_file = os.path.join(save_path, "nnUnet_BAGLS_1077.png")
#output_file = os.path.join(save_path, "nnUnet_Swallowing_0153.png")
output_file = os.path.join(save_path, "nnUnet_BrainMeningioma_0275.png")

if not os.path.exists(save_path):
    os.makedirs(save_path)

nii_image = nib.load(nii_file_path)
nii_data = nii_image.get_fdata()

# Check 2D and normalize the data to 0-255 for PNG format
if nii_data.ndim == 2:  # Confirming it's 2D
    # Normalize the data to the range 0-255 and convert to uint8
    nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255
    nii_data = nii_data.astype(np.uint8)

    # Convert to PIL Image for resizing, rotating, and flipping
    img = Image.fromarray(nii_data)

    # Resize to 256x256
    img = img.resize((256, 256), Image.ANTIALIAS)

    # Rotate 90 degrees to the right and flip horizontally
    img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)

    # Save as PNG
    img.save(output_file)
    print(f"Saved image as {output_file}")
else:
    print("Error: The loaded .nii.gz file is not 2D.")
