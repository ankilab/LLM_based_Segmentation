import torch
import os
import sys
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from skimage.filters import threshold_otsu

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

#model_dir = "D:\\qy44lyfe\\LLM segmentation\github repo\\LLM_based_Segmentation\\Modells 2025\\Claude 4 Sonnet"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\DeepSeek R1"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\DeepSeek V3"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Gemini 2.5 pro"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\GPT o3"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\GPT o4-mini-high"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Grok 3"
model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Grok 3 mini_reasoning"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Llama 4 Maverick"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Mistral Medium 3"
#model_dir = "D:\\qy44lyfe\\LLM segmentation\\github repo\\LLM_based_Segmentation\\Modells 2025\\Qwen 3_235B"

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

    #mask_name = img_name                               # bolus dataset
    mask_name = img_name.replace('.png', '_seg.png')  # BAGLS dataset
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

# OLD visalization func ##############################################################
# def visualize_prediction(input_img, ground_truth, prediction, img_name, dice_score, save_path):
#     fig, axs = plt.subplots(1, 3, figsize=(12, 5))
#
#     # Squeeze the first dimension to convert (1, 256, 256) -> (256, 256)
#     axs[0].imshow(input_img[0].squeeze(), cmap='gray')
#     axs[0].set_title(f"Input Image: {img_name}", fontsize=12)
#     axs[0].axis('off')
#
#     axs[1].imshow(ground_truth[0].squeeze(), cmap='gray')
#     axs[1].set_title("Ground Truth", fontsize=12)
#     axs[1].axis('off')
#
#     axs[2].imshow(prediction[0].squeeze(), cmap='gray')
#     axs[2].set_title(f"Prediction\nDice Score: {dice_score:.4f}", fontsize=12)
#     axs[2].axis('off')
#
#     # Use tight layout and save the figure
#     plt.tight_layout()
#
#     # Extract model name from the last part of model_dir
#     model_name = os.path.basename(model_dir.strip("/\\"))  # Strips trailing slashes and extracts the last part of the path
#
#     # Save the figure with 600 DPI and the new file name format
#     output_filename = f"{model_name}_{img_name}_prediction.png"
#     output_filepath = os.path.join(save_path, output_filename)
#
#     plt.savefig(output_filepath, dpi=600)
#     # plt.show()


# Main function for inference and visualization (OLD) #########################################################
# def infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path, image_size=(256, 256)):
#     # Define and load the model architecture
#     model = UNet()  # UNet is now dynamically imported from 'model.py'
#     #model = UNet(n_channels=1, n_classes=1)  # UNet is now dynamically imported from 'model.py'
#     model.load_state_dict(torch.load(state_dict_path))  # Load the state dictionary
#
#     # # load checkpoint as weights =========================
#     # ckpt = torch.load(state_dict_path, map_location='cpu')
#     # print("CKPT keys:", ckpt.keys())
#     # model.load_state_dict(ckpt['model_state_dict'])
#     # # ====================================================
#
#     # load & strip module. prefix if needed
#     #================================================
#     # sd = torch.load(state_dict_path, map_location='cpu')
#     # #print("CKPT keys:", sd.keys())
#     # if any(k.startswith("module.") for k in sd):
#     #     stripped = {k[len("module."):]: v for k, v in sd.items()}
#     # else:
#     #     stripped = sd
#     # model.load_state_dict(stripped)
#     #=====================================================
#
#     model.eval()
#
#     # Load input image and ground truth mask
#     image, ground_truth = load_image_and_mask(image_path, mask_dir, image_size)
#     image = image.unsqueeze(0)  # Add batch dimension
#
#     with torch.no_grad():
#         # Forward pass
#         prediction = model(image)
#         # Resize the predicted mask to match the size of the ground truth if necessary
#         if prediction.shape != ground_truth.unsqueeze(0).shape:
#             prediction = F.interpolate(prediction, size=ground_truth.shape[1:], mode='bilinear', align_corners=False)
#         # Binarize the predictions
#         prediction = (prediction > 0.5).float()
#
#     # OR TRY:
#     # with torch.no_grad():
#     #     logits = model(image)
#     #     if logits.shape != ground_truth.unsqueeze(0).shape:
#     #         logits = F.interpolate(logits,
#     #                                size=ground_truth.shape[1:],
#     #                                mode='bilinear',
#     #                                align_corners=False)
#     #     probs      = torch.sigmoid(logits)
#     #     prediction = (probs > 0.5).float()
#
#     # Calculate Dice score
#     dice_score = dice_coefficient(prediction, ground_truth).item()
#     # Visualize the results
#     visualize_prediction(image, ground_truth, prediction, img_name, dice_score, save_path)

# NEW
def visualize_prediction(input_img, ground_truth, prediction, img_name, dice_score, save_path):
    """
    input_img:  tensor [1, C, H, W]
    ground_truth, prediction: [1, 1, H, W]
    """
    img = input_img[0].cpu()
    C = img.shape[0]

    # convert to HxW or HxWx3 for imshow
    if C == 1:
        img_disp = img.squeeze(0).numpy()
        cmap0 = 'gray'
    else:
        # [C,H,W] -> [H,W,C]
        img_disp = img.permute(1, 2, 0).numpy()
        cmap0 = None

    gt_disp = ground_truth[0,0].cpu().numpy()
    pred_disp = prediction[0,0].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].imshow(img_disp, cmap=cmap0)
    axs[0].set_title(f"Input: {img_name}", fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(gt_disp, cmap='gray')
    axs[1].set_title("Ground Truth", fontsize=12)
    axs[1].axis('off')

    axs[2].imshow(pred_disp, cmap='gray')
    axs[2].set_title(f"Prediction\nDice: {dice_score:.4f}", fontsize=12)
    axs[2].axis('off')

    plt.tight_layout()
    model_name = os.path.basename(model_dir.strip("/\\"))
    out_fn = f"{model_name}_{img_name}_prediction_new1.png"
    plt.savefig(os.path.join(save_path, out_fn), dpi=600)
    plt.close(fig)

## NEW
## for claude, DeepSeek, Gemini, GPTs, Llama, mistral, Qwen,
# def infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path, image_size=(256, 256)):
#     import torch.nn.functional as F
#
#     #mask_name = img_name                               # bolus dataset
#     mask_name = img_name.replace('.png', '_seg.png')  # BAGLS dataset
#     #mask_name = img_name.replace('.jpg', '_m.jpg')   # Brain tumor dataset
#
#     # 1) load checkpoint
#     sd = torch.load(state_dict_path, map_location='cpu')
#     # strip DataParallel prefix if present
#     if any(k.startswith("module.") for k in sd):
#         sd = {k.replace("module.", ""): v for k, v in sd.items()}
#
#     # 2) inspect first conv and last conv to figure in_ch, n_classes
#     conv_keys = [k for k,v in sd.items() if isinstance(v, torch.Tensor) and v.ndim==4]
#     first_w   = sd[conv_keys[0]]    # [out_ch, in_ch, k, k]
#     last_w    = sd[conv_keys[-1]]   # [n_classes, feat, 1,1]
#     in_ch     = first_w.shape[1]
#     n_classes = last_w.shape[0]
#
#     # 3) instantiate matching UNet, load all matching weights
#     model = UNet(in_ch, n_classes)
#     msd   = model.state_dict()
#     filtered = {k:v for k,v in sd.items() if k in msd and v.shape==msd[k].shape}
#     msd.update(filtered)
#     model.load_state_dict(msd)
#     model.eval()
#
#     # 4) load + resize + to-tensor the grayscale image
#     pil = Image.open(image_path).convert("L")
#     pil = transforms.Resize(image_size)(pil)
#     img_t = TF.to_tensor(pil).unsqueeze(0)  # [1,1,H,W]
#
#     # 5) tile if UNet expects 3 channels
#     if in_ch == 1:
#         inp = img_t
#     elif in_ch == 3:
#         inp = img_t.repeat(1,3,1,1)
#     else:
#         raise RuntimeError(f"Model expects {in_ch} input channels; only 1 or 3 are supported")
#
#     # 6) load + resize + to-tensor the mask
#     mp  = Image.open(os.path.join(mask_dir, mask_name)).convert("L")
#     mp  = transforms.Resize(image_size)(mp)
#     mask_t = TF.to_tensor(mp).unsqueeze(0)
#     mask_t = (mask_t > 0.5).float()
#
#     # 7) forward, slice to single channel, resize if needed
#     with torch.no_grad():
#         out = model(inp)                # [1,C,H,W]
#         if out.shape[1] != 1:
#             out = out[:, :1, :, :]      # keep only channel 0
#         if out.shape != mask_t.shape:
#             out = F.interpolate(
#                 out,
#                 size=mask_t.shape[2:],
#                 mode="bilinear",
#                 align_corners=False
#             )
#         pred = (out > 0.5).float()
#
#     # 8) dice + visualize
#     d = dice_coefficient(pred, mask_t).item()
#     visualize_prediction(inp, mask_t, pred, img_name, d, save_path)


## NEW: for grok 3, grok 3 mini ##########################################################################
def infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path, image_size=(256, 256)):
    # 1. build model
    model = UNet()
    sd = torch.load(state_dict_path, map_location='cpu')
    # load state_dict (bare or checkpoint)
    if 'model_state_dict' in sd:
        model.load_state_dict(sd['model_state_dict'])
    else:
        model.load_state_dict(sd)
    model.eval()

    # 2. load image+mask
    img_t, gt_t = load_image_and_mask(image_path, mask_dir, image_size)
    img_t = img_t.unsqueeze(0)   # [1,C,H,W]
    gt_t  = gt_t.unsqueeze(0)    # [1,1,H,W]

    # 3. inference
    with torch.no_grad():
        logits = model(img_t)    # [1, C_out, H, W]
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1:2]

        pmin, pmax = probs.min().item(), probs.max().item()
        print(f" prob range: {pmin:.4f} .. {pmax:.4f}")

        # 4. compute Otsu threshold on CPU numpy array
        prob_np = probs[0,0].cpu().numpy()
        try:
            otsu_thresh = threshold_otsu(prob_np)
        except ValueError:
            # fall back if uniform
            otsu_thresh = 0.5
        print(f" Otsu threshold: {otsu_thresh:.4f}")

        pred_t = (probs > otsu_thresh).float()

    # 5. dice
    dice_score = dice_coefficient(pred_t, gt_t).item()
    print(" Dice:", dice_score)

    # 6. visualize + save
    visualize_prediction(img_t, gt_t, pred_t, img_name, dice_score, save_path)




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
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Claude 4 Sonnet\\out of the box\\BAGLS output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek R1\\out of the box\\BAGLS output\\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek V3\\out of the box\\BAGLS output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Gemini 2.5 Pro\\out of the box\\BAGLS output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\\BAGLS output\\unet_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o4-mini-high\\out of the box\\BAGLS output\\unet_model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3\\out of the box\\BAGLS output\\model_state_dict.pth"
    state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3 mini Reasoning\\out of the box\\BAGLS output\\model_state_dict.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Llama 4 Maverick\\out of the box\\BAGLS output\\model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Mistral Medium 3\\out of the box\\BAGLS output\\mask fix\\unet_model.pth"
    #state_dict_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Qwen 3_235B\\out of the box\\BAGLS output\\model_state.pth"

    #=================================================================================================



    image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"  # Folder containing input images
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Skin cancer\\subset\\images"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_images\\converted"
    #image_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\images"

    mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset" # Folder containing mask images
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Swallowing\\masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Brain Meningioma\\Masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\Skin cancer\\subset\\masks"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\DRIVE\\combined_masks\\converted"
    #mask_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\utrine myoma converted\\masks_myoma"

    img_name = '1077.png'  # BAGLS
    #img_name = '153.png'   # bolus
    #img_name = 'Tr-me_0275.jpg'  # Brain tumor
    #img_name = 'ISIC_0024306.png'  # skin cancer
    #img_name = '11_g_HRF.jpg'  # Retina
    #img_name = 'case_0039_UMD_221129_055_slice013.png'  # Myoma


    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\Inference\\inf_2025\\'

    image_path = os.path.join(image_dir, img_name)

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run inference on the selected image
    infer_single_image(state_dict_path, image_path, mask_dir, img_name, save_path)
