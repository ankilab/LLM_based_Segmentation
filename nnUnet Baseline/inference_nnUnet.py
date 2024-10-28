import torch
import os
import numpy as np
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.models import nnUNet
from nnunetv2.paths import nnUNet_results, nnUNet_preprocessed
from nnunetv2.configuration import default_plans_identifier

# Define paths and parameters
checkpoint_path = "path_to_your_model/model.pth"  # Path to your .pth file containing only the weights
task_name = "TaskXXX_YourTaskName"  # Replace with the task name you trained with, e.g., "Task001_BrainTumor"
configuration = "2d"  # or "3d_fullres" depending on your model configuration
output_dir = "path_to_save_predictions"  # Directory where predictions will be saved as .png
raw_image_path = "path_to_unseen_image.nii.gz"  # Path to the single unseen raw image

# Step 1: Load the checkpoint and extract the network weights
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['network_weights']  # Extract the model weights

# Step 2: Initialize the nnUNetPredictor with nnU-Net model
predictor = nnUNetPredictor(
    task_name=task_name,
    configuration=configuration,
    plans_identifier=default_plans_identifier,
    output_folder=output_dir,
)

# Load the extracted weights into the nnU-Net model
predictor.model.load_state_dict(state_dict)
predictor.model.eval()

# Step 3: Run inference and get prediction as a numpy array
with torch.no_grad():
    predicted_mask = predictor.predict_single_image(raw_image_path)
    predicted_mask = predicted_mask.squeeze()  # Remove any singleton dimensions

# Step 4: Convert prediction to binary (optional, depending on your segmentation)
predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Binarize and scale to 0-255

# Step 5: Save the prediction as a .png file
output_png_path = os.path.join(output_dir, "prediction.png")
Image.fromarray(predicted_mask).save(output_png_path)

print(f"Prediction saved to: {output_png_path}")
