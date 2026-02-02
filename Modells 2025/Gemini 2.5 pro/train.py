# unet_segmentation/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# --- Loss Function ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice


# --- Metrics ---
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()  # Binarize

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()


# --- Training, Validation, and Testing Functions ---

def train_epoch(loader, model, optimizer, loss_fn_bce, loss_fn_dice, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training")
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss_bce = loss_fn_bce(outputs, masks)
        loss_dice = loss_fn_dice(outputs, masks)
        loss = loss_bce + loss_dice  # Combined loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate_epoch(loader, model, loss_fn_bce, loss_fn_dice, device):
    model.eval()
    running_loss = 0.0
    batch_dice_scores = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss_bce = loss_fn_bce(outputs, masks)
            loss_dice = loss_fn_dice(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            dice = dice_coefficient(outputs, masks)
            batch_dice_scores.append(dice)

            pbar.set_postfix(loss=loss.item(), dice=dice)

    avg_loss = running_loss / len(loader)
    return avg_loss, batch_dice_scores


def test_model(loader, model, device):
    model.eval()
    batch_dice_scores = []

    pbar = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            batch_dice_scores.append(dice)
            pbar.set_postfix(dice=dice)

    return batch_dice_scores


# --- Visualization and Saving Functions ---

def save_losses(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)

    df_train = pd.DataFrame({'Epoch': epochs, 'Loss': train_losses})
    df_train.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)

    df_val = pd.DataFrame({'Epoch': epochs, 'Loss': val_losses})
    df_val.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)


def save_dice_scores(all_epoch_dice, save_path, filename="validation_dice_scores.xlsx"):
    max_batches = max(len(d) for d in all_epoch_dice)
    dice_data = {f'Batch_{i + 1}': [epoch_dice[i] if i < len(epoch_dice) else None for epoch_dice in all_epoch_dice]
                 for i in range(max_batches)}
    df_dice = pd.DataFrame(dice_data)
    df_dice.index.name = 'Epoch'
    df_dice.index = df_dice.index + 1
    df_dice.to_excel(os.path.join(save_path, filename))


def plot_and_save_losses(save_path):
    df_train = pd.read_excel(os.path.join(save_path, 'train_losses.xlsx'))
    df_val = pd.read_excel(os.path.join(save_path, 'val_losses.xlsx'))

    plt.figure(figsize=(10, 5))
    plt.plot(df_train['Epoch'], df_train['Loss'], label='Training Loss')
    plt.plot(df_val['Epoch'], df_val['Loss'], label='Validation Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()


def visualize_predictions(model, loader, device, save_path, num_samples=5):
    model.eval()

    # Get random samples from the loader's dataset
    indices = np.random.choice(len(loader.dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle('Input Image | Ground Truth | Prediction', fontsize=16)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, img_name = loader.dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            # Prediction
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze(0)

            # Prepare for plotting
            image_to_plot = image.permute(1, 2, 0).numpy()
            mask_to_plot = mask.permute(1, 2, 0).numpy()
            pred_to_plot = pred_mask.permute(1, 2, 0).numpy()

            # Row i, Column 0: Input Image
            axes[i, 0].imshow(image_to_plot, cmap='gray')
            axes[i, 0].set_title(f"Input: {img_name}")
            axes[i, 0].axis('off')

            # Row i, Column 1: Ground Truth
            axes[i, 1].imshow(mask_to_plot, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            # Row i, Column 2: Prediction
            axes[i, 2].imshow(pred_to_plot, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_path, 'test_predictions.png'))
    plt.close()