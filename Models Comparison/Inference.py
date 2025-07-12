import torch
import os
import sys
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# Dynamically adding the path to 'model.py'
## 2024 models: ==============================================================================
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\Bing Microsoft Copilot" # The folder the model.py is located
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\Claude 3.5 Sonnet"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\Copilot"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\Gemini 1.5 Pro"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\GPT o1 preview"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\GPT4"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\GPT4o"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2024\LLaMA 3.1_405B"
#================================================================================================

## 2025 models: ================================================================================

#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Claude 4 Sonnet"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\DeepSeek R1"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\DeepSeek V3"
model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Gemini 2.5 pro"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\GPT o3"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\GPT o4-mini-high"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Grok 3"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Grok 3 mini_reasoning"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Llama 4 Maverick"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Mistral Medium 3"
#model_dir = "D:\qy44lyfe\LLM segmentation\github repo\LLM_based_Segmentation\Modells 2025\Qwen 3_235B"

#=================================================================================================
sys.path.append(model_dir)

# Dynamically importing 'model.py'
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

    mask_name = img_name                               # bolus dataset
    # mask_name = img_name.replace('.png', '_seg.png')  # BAGLS dataset
    #mask_name = img_name.replace('.jpg', '_m.jpg')   # Brain tumor dataset

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


def visualize_prediction(input_img, ground_truth, prediction, img_name, dice_score, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Squeeze the first dimension to convert (1, 256, 256) -> (256, 256)
    axs[0].imshow(input_img[0].squeeze(), cmap='gray')
    axs[0].set_title(f"Input Image: {img_name}", fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(ground_truth[0].squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth", fontsize=12)
    axs[1].axis('off')

    axs[2].imshow(prediction[0].squeeze(), cmap='gray')
    axs[2].set_title(f"Prediction\nDice Score: {dice_score:.4f}", fontsize=12)
    axs[2].axis('off')

    # Use tight layout and save the figure
    plt.tight_layout()

    # Extract model name from the last part of model_dir
    model_name = os.path.basename(model_dir.strip("/\\"))  # Strips trailing slashes and extracts the last part of the path

    # Save the figure with 600 DPI and the new file name format
    output_filename = f"{model_name}_{img_name}_prediction.png"
    output_filepath = os.path.join(save_path, output_filename)

    plt.savefig(output_filepath, dpi=600)
    # plt.show()


# Main function for inference and visualization
def infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path, image_size=(256, 256)):
    # Define and load the model architecture
    #model = UNet()  # UNet is now dynamically imported from 'model.py'
    model = UNet(n_channels=1, n_classes=1)  # UNet is now dynamically imported from 'model.py'
    model.load_state_dict(torch.load(state_dict_path))  # Load the state dictionary

    # # load checkpoint as weights =========================
    # ckpt = torch.load(state_dict_path, map_location='cpu')
    # print("CKPT keys:", ckpt.keys())
    # model.load_state_dict(ckpt['model_state_dict'])
    # # ====================================================

    model.eval()

    # Load input image and ground truth mask
    image, ground_truth = load_image_and_mask(image_path, mask_dir, image_size)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Forward pass
        prediction = model(image)

        # Resize the predicted mask to match the size of the ground truth if necessary
        if prediction.shape != ground_truth.unsqueeze(0).shape:
            prediction = F.interpolate(prediction, size=ground_truth.shape[1:], mode='bilinear', align_corners=False)

        # Binarize the predictions
        prediction = (prediction > 0.5).float()

    # Calculate Dice score
    dice_score = dice_coefficient(prediction, ground_truth).item()

    # Visualize the results
    visualize_prediction(image, ground_truth, prediction, img_name, dice_score, save_path)

if __name__ == "__main__":
    # Paths to model, input image, and ground truth mask
    ## 2024 models =======================================================================================
    # BAGLS
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\GPT o1 preview\\out of the box\\utrine myoma output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\GPT 4\\out of the box\\Retina output\\model_state.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\GPT 4o\\out of the box\\utrine myoma output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Copilot\\out of the box\\utrine myoma output\\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Bing Microsoft Copilot\\out of the box\\utrine myoma output\\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Claude 3.5 Sonnet\\out of the box\\utrine myoma output\\best_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Gemini 1.5 pro\\out of the box\\utrine myoma output\\unet_model_weights.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\LLAMA 3.1 405B\\out of the box\\utrine myoma output\\unet_model_state_dict.pth"

    ## 2025 models =======================================================================================
    # BAGLS
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Claude 4 Sonnet\\out of the box\\Myoma output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek R1\\out of the box\\Myoma output\\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek V3\\out of the box\\Myoma output\\model_state_dict.pth"
    state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Gemini 2.5 Pro\\out of the box\\Myoma output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\Myoma output\\unet_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o4-mini-high\\out of the box\\Myoma output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3\\out of the box\Myoma output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3 mini Reasoning\\out of the box\\Myoma output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Llama 4 Maverick\\out of the box\\Myoma output\\model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Mistral Medium 3\\out of the box\\Myoma output\\mask fix\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Qwen 3_235B\\out of the box\\Myoma output\\model_state.pth"

    #=================================================================================================



    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"  # Folder containing input images
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Skin cancer\\subset\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_images\\converted"
    image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\images"

    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset" # Folder containing mask images
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\Masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Skin cancer\\subset\\masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_masks\\converted"
    mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\masks_myoma"

    #img_name = '1077.png'  # BAGLS
    #img_name = '153.png'   # bolus
    #img_name = 'Tr-me_0275.jpg'  # Brain tumor
    #img_name = 'ISIC_0024306.png'  # skin cancer
    #img_name = '11_g_HRF.jpg'  # Retina
    img_name = 'case_0039_UMD_221129_055_slice013.png'  # Myoma


    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\Inference\\inf_2025\\'

    image_path = os.path.join(image_dir, img_name)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run inference on the selected image
    infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path)
