import torch
import os
import sys
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# Dynamically add the path to 'model.py'
model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\GPT o1 preview"  # The folder where your model.py is located
sys.path.append(model_dir)

# Dynamically import 'model.py'
model_module = importlib.import_module('model')  # Import 'model.py' as a module
UNet = model_module.UNet  # Access the UNet class from the module

# Dice coefficient function
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

# Function to load a single image and mask, resize and return tensors
def load_image_and_mask(image_path, mask_dir, image_size=(256, 256)):
    img_name = os.path.basename(image_path)  # Get the image filename (e.g., '11.png')
    mask_name = img_name.replace('.png', '_seg.png')  # Replace '.png' with '_seg.png' for the mask

    mask_path = os.path.join(mask_dir, mask_name)  # Full path to the mask

    # Load image and mask
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    mask = Image.open(mask_path).convert('L')    # Convert to grayscale

    # Resize images and masks
    resize = transforms.Resize(image_size)
    image = resize(image)
    mask = resize(mask)

    # Convert to tensors
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    mask = (mask > 0.5).float()  # Binarize the mask
    return image, mask

# Function to visualize the input, ground truth mask, and predicted mask
def visualize_prediction(input_img, ground_truth, prediction, img_name, dice_score, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Squeeze the first dimension to convert (1, 256, 256) -> (256, 256)
    axs[0].imshow(input_img[0].squeeze(), cmap='gray')  # Remove the first dimension for display
    axs[0].set_title(f"Input Image: {img_name}")
    axs[0].axis('off')

    axs[1].imshow(ground_truth[0].squeeze(), cmap='gray')  # Remove the first dimension for display
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(prediction[0].squeeze(), cmap='gray')  # Remove the first dimension for display
    axs[2].set_title(f"Prediction\nDice Score: {dice_score:.4f}")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{img_name}_prediction.png'))
    plt.show()

# Main function for inference and visualization
def infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path, image_size=(256, 256)):
    # Define and load the model architecture
    model = UNet()  # UNet is now dynamically imported from 'model.py'
    model.load_state_dict(torch.load(state_dict_path))  # Load the state dictionary
    model.eval()

    # Load input image and ground truth mask
    image, ground_truth = load_image_and_mask(image_path, mask_dir, image_size)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Forward pass
        prediction = model(image)
        prediction = (prediction > 0.5).float()  # Binarize predictions

    # Calculate Dice score
    dice_score = dice_coefficient(prediction, ground_truth).item()

    # Visualize the results
    visualize_prediction(image, ground_truth, prediction, img_name, dice_score, save_path)

if __name__ == "__main__":
    # Paths to model, input image, and ground truth mask
    state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\GPT o1 preview\\out of the box\\BAGLS output\\unet_model_state_dict.pth"
    image_dir = 'D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset'  # Folder containing input images
    mask_dir = 'D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset'  # Folder containing mask images
    img_name = '1077.png'  # The name of the image you want to visualize (e.g., '11.png')
    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison'  # Directory where you want to save the output

    image_path = os.path.join(image_dir, img_name)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run inference on the selected image
    infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path)
