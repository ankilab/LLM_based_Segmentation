# train.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, leave=False, desc='Training')
    for inputs, masks, _ in loop:
        inputs = inputs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        loop.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_model(model, dataloader, criterion, device, epoch_num, save_path):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    loop = tqdm(dataloader, leave=False, desc='Validation')
    with torch.no_grad():
        for inputs, masks, _ in loop:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * inputs.size(0)

            # Calculate Dice score per batch
            preds = (outputs > 0.5).float()
            dice = dice_coefficient(preds, masks)
            dice_scores.append(dice.item())

            loop.set_postfix({'loss': loss.item(), 'dice': dice.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

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

    return epoch_loss

def test_model(model, dataloader, device, save_path):
    model.eval()
    dice_scores = []
    all_inputs = []
    all_masks = []
    all_preds = []
    all_img_names = []
    loop = tqdm(dataloader, leave=False, desc='Testing')
    with torch.no_grad():
        for inputs, masks, img_names in loop:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)

            preds = (outputs > 0.5).float()
            dice = dice_coefficient(preds, masks)
            dice_scores.append(dice.item())

            loop.set_postfix({'dice': dice.item()})

            all_inputs.extend(inputs.cpu())
            all_masks.extend(masks.cpu())
            all_preds.extend(preds.cpu())
            all_img_names.extend(img_names)

    # Save dice scores to Excel
    df_new = pd.DataFrame([dice_scores])
    excel_path = os.path.join(save_path, 'test_dice_scores.xlsx')
    df_new.to_excel(excel_path, index=False, header=False)

    return all_inputs, all_masks, all_preds, all_img_names

def save_losses_to_excel(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))

    # Train losses
    df_train = pd.DataFrame([epochs, train_losses])
    df_train.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False, header=False)

    # Validation losses
    df_val = pd.DataFrame([epochs, val_losses])
    df_val.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False, header=False)

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

def visualize_predictions(inputs, masks, preds, img_names, save_path):
    indices = random.sample(range(len(inputs)), 5)
    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    for i, idx in enumerate(indices):
        input_img = inputs[idx][0]  # Get the first channel
        mask_img = masks[idx][0]
        pred_img = preds[idx][0]
        img_name = img_names[idx]

        axs[i, 0].imshow(input_img, cmap='gray')
        axs[i, 0].set_title(f"Input Image: {img_name}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask_img, cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(pred_img, cmap='gray')
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_predictions.png'))
    plt.close()
