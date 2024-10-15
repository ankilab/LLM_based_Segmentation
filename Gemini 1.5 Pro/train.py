import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import torch.nn.functional as F  # Import F for interpolation

# def dice_coef(preds, targets, smooth=1.):
#     preds = (preds > 0.5).float()  # Convert probabilities to binary predictions
#     intersection = (preds * targets).sum()
#     dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
#     return dice.item()

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

    for i, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        resized_masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')  # Resize masks
        loss = criterion(outputs, resized_masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def validate(model, val_loader, criterion, device, epoch, save_path):
    model.eval()
    total_loss = 0
    total_dice = 0
    dice_scores = []
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        for batch_idx, (images, masks) in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            resized_masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            loss = criterion(outputs, resized_masks)
            total_loss += loss.item()

            outputs = (outputs > 0.5).float()
            dice = dice_coeff(outputs, resized_masks)
            dice_scores.append(dice.cpu().item())
            total_dice += dice  # Directly accumulate the Dice score. No need for .item()

            loop.set_postfix(loss=loss.item(), dice=dice) # No .item() here either

        loop.set_description(f"Validation Epoch {epoch}")

    val_loss = total_loss / len(val_loader)
    val_dice = total_dice / len(val_loader)
    # Save dice scores to Excel
    df_new = pd.DataFrame([dice_scores])
    excel_path = os.path.join(save_path, 'validation_dice_scores.xlsx')
    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False, header=False)
    else:
        # Read existing data
        df_existing = pd.read_excel(excel_path, header=None)
        # Append new data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Write back to Excel
        df_combined.to_excel(excel_path, index=False, header=False)

    return val_loss, val_dice


def test(model, test_loader, device, save_path):
    model.eval()
    total_dice = 0
    dice_scores = []
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (images, masks) in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Resize outputs to match masks' dimensions using interpolation:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)


            outputs = (outputs > 0.5).float()  # Apply thresholding *after* resizing
            dice = dice_coeff(outputs, masks)
            total_dice += dice
            dice_scores.append(dice.cpu().item())

            loop.set_postfix(dice=dice.cpu().item()) # Move to CPU before getting item

        loop.set_description(f"Testing")

    avg_dice = total_dice / len(test_loader)
    # Save dice scores to Excel
    df_new = pd.DataFrame([dice_scores])
    excel_path = os.path.join(save_path, 'test_dice_scores.xlsx')
    df_new.to_excel(excel_path, index=False, header=False)

    return avg_dice.cpu().item() # Move to CPU before returning


# def visualize_predictions(model, test_loader, device, save_path, num_samples=5):
#     model.eval()
#     indices = np.random.choice(len(test_loader.dataset), num_samples, replace=False)
#     fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
#
#     with torch.no_grad():
#         for i, idx in enumerate(indices):
#             image, mask = test_loader.dataset[idx]
#             image_tensor = image.unsqueeze(0).to(device)
#             prediction = model(image_tensor)
#             prediction = (prediction > 0.5).float() #binary
#
#             image = image.cpu().numpy().squeeze()
#             mask = mask.cpu().numpy().squeeze()
#             prediction = prediction.cpu().numpy().squeeze()
#
#             file_id = test_loader.dataset.image_files[idx].split('.')[0]
#
#             axes[i, 0].imshow(image, cmap='gray')
#             axes[i, 0].set_title(f"{file_id}\nInput Image")
#             axes[i, 0].axis('off')
#
#             axes[i, 1].imshow(mask, cmap='gray')
#             axes[i, 1].set_title(f"{file_id}\nGround Truth")
#             axes[i, 1].axis('off')
#
#             axes[i, 2].imshow(prediction, cmap='gray')
#             axes[i, 2].set_title(f"{file_id}\nPrediction")
#             axes[i, 2].axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "predictions_visualization.png"))


def visualize_predictions(model, dataloader, device, save_path):
    model.eval()
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    axs = axs.ravel()

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i == 5:
                break
            # Ensure images are 4D
            images = images.to(device)
            masks = masks.to(device)

            # Ensure images have the batch dimension (if not already)
            if images.dim() == 3:
                images = images.unsqueeze(0)

            outputs = model(images)
            outputs = (outputs > 0.5).float()

            # Plot input image, ground truth, and prediction
            axs[i * 3].imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 1].imshow(masks[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 2].imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')

    plt.savefig(f'{save_path}/predictions.png')
    plt.close()

#
# def plot_losses(train_losses, val_losses, save_path):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Losses')
#     plt.legend()
#     plt.savefig(os.path.join(save_path, "loss_plot.png"))

def plot_losses(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'orange', label='Validation loss')
    plt.title('Training and Validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))
    plt.close()