import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# Loss function with background weighting
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.7):
        super(DiceBCELoss, self).__init__()
        self.weight = weight  # Weight for BCE (Dice weight = 1 - weight)

    def forward(self, inputs, targets):
        # Binary Cross Entropy
        bce = nn.functional.binary_cross_entropy(inputs, targets)

        # Dice Coefficient
        smooth = 1.0
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return self.weight * bce + (1 - self.weight) * (1 - dice)


def dice_score(preds, targets):
    preds_bin = (preds > 0.5).float()
    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    return (2. * intersection) / (preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)))


def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate Dice per sample
            batch_dice = dice_score(outputs, masks).cpu().numpy()
            dice_scores.extend(batch_dice)

    return running_loss / len(loader.dataset), np.array(dice_scores)


def test_model(model, loader, device):
    model.eval()
    dice_scores = []
    all_images, all_masks, all_preds, all_names = [], [], [], []

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            batch_dice = dice_score(outputs, masks).cpu().numpy()
            dice_scores.extend(batch_dice)

            # Store samples for visualization
            if len(all_images) < 5:
                all_images.append(images.cpu())
                all_masks.append(masks.cpu())
                all_preds.append(outputs.cpu())
                all_names.extend(names)

    # Concatenate stored samples
    vis_images = torch.cat(all_images, dim=0)[:5]
    vis_masks = torch.cat(all_masks, dim=0)[:5]
    vis_preds = torch.cat(all_preds, dim=0)[:5]

    return np.array(dice_scores), (vis_images, vis_masks, vis_preds, all_names[:5])


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()


def save_sample_visualization(samples, save_path):
    images, masks, preds, names = samples
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    fig.suptitle('Segmentation Results', fontsize=16)

    for i in range(5):
        # Input image (denormalize)
        img = images[i].squeeze().numpy()
        img = (img * 0.5) + 0.5  # [-1,1] to [0,1]

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Input: {names[i]}")

        # Ground truth mask
        mask = masks[i].squeeze().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")

        # Prediction
        pred = preds[i].squeeze().numpy()
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sample_predictions.png'))
    plt.close()