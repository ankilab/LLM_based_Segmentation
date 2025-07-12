import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Dice coefficient function
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

# Function to load the input image, ground truth mask, and saved prediction
def load_image_and_mask(image_path, mask_dir, prediction_path, image_size=(256, 256)):
    img_name = os.path.basename(image_path)  # Get the image filename (e.g., '11.png')

    # Load input image and mask
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    #image = Image.open(image_path) # keep RGB

    mask_name = img_name                               # bolus dataset
    #mask_name = img_name.replace('.png', '_seg.png')  # BAGLS dataset
    #mask_name = img_name.replace('.jpg', '_m.jpg')   # Brain tumor dataset

    mask_path = os.path.join(mask_dir, mask_name)
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale

    # Load the saved prediction
    prediction = Image.open(prediction_path).convert('L')  # Convert to grayscale

    # Resize images and masks
    resize = transforms.Resize(image_size)
    image = resize(image)
    mask = resize(mask)
    prediction = resize(prediction)

    # Convert to tensors
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    prediction = TF.to_tensor(prediction)

    # Binarize mask and prediction
    mask = (mask > 0.5).float()
    prediction = (prediction > 0.5).float()

    return image, mask, prediction

# Visualization function
def visualize_prediction(input_img, ground_truth, prediction, img_name, dice_score, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Display input image (gray)
    axs[0].imshow(input_img.squeeze(), cmap='gray')

    ## FOR COLOR IMAGE: =====================================================
    # # move channels to the last dimension, then show in full color
    # # For color image: convert the tensor to HWC and then show
    # img_tensor = input_img.squeeze()  # shape = (3, H, W)
    # # if it’s a torch.Tensor, use permute + .numpy()
    # try:
    #     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # except AttributeError:
    #     # already a NumPy array?
    #     img_np = np.moveaxis(img_tensor, 0, -1)
    # axs[0].imshow(img_np)
    ##========================================================================

    axs[0].set_title(f"Input Image: {img_name}", fontsize=12)
    axs[0].axis('off')

    # Display ground truth mask
    axs[1].imshow(ground_truth.squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth", fontsize=12)
    axs[1].axis('off')

    # Display prediction mask with Dice score
    axs[2].imshow(prediction.squeeze(), cmap='gray')
    axs[2].set_title(f"Prediction\nDice Score: {dice_score:.4f}", fontsize=12)
    axs[2].axis('off')

    # Use tight layout and save the figure
    plt.tight_layout()
    output_filename = f"{img_name}_nnUnet_Baseline_comparison.png"
    output_filepath = os.path.join(save_path, output_filename)
    plt.savefig(output_filepath, dpi=600)
    #plt.show()

# Main function to load and compare prediction with ground truth
def compare_prediction_and_mask(image_path, mask_dir, prediction_path, img_name, save_path, image_size=(256, 256)):
    # Load input image, ground truth mask, and saved prediction
    image, ground_truth, prediction = load_image_and_mask(image_path, mask_dir, prediction_path, image_size)

    # Resize the predicted mask to match the size of the ground truth if necessary
    if prediction.shape != ground_truth.shape:
        prediction = F.interpolate(prediction.unsqueeze(0), size=ground_truth.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

    # Binarize the resized prediction
    prediction = (prediction > 0.5).float()

    # Calculate Dice score
    dice_score = dice_coefficient(prediction, ground_truth).item()

    # Visualize the results
    visualize_prediction(image, ground_truth, prediction, img_name, dice_score, save_path)

if __name__ == "__main__":
    # Paths for the original image, ground truth mask, and saved prediction
    # BAGLS
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"  # Folder containing input images
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Skin cancer\\subset\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_images\\converted"
    image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\images"

    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset" # Folder containing mask images
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\Masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_masks\\converted"
    mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\masks_myoma"

    #prediction_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\nnUnet_BAGLS_1077.png"
    #prediction_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\nnUnet_Swallowing_0153.png"
    #prediction_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\nnUnet_BrainMeningioma_0275.png"
    #prediction_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\predictions\\nnUnet_ISIC_0024306.png"
    prediction_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\nnUnet Baseline\\predictions\\nnUnet_case_0039_UMD_221129_055_slice013.png"


    img_name = 'case_0039_UMD_221129_055_slice013.png'   # Example image name

    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\Inference\\'  # Directory to save the output
    image_path = os.path.join(image_dir, img_name)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run comparison on the selected image
    compare_prediction_and_mask(image_path, mask_dir, prediction_path, img_name, save_path)
