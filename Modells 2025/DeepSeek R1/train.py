import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchinfo import summary


# Loss function with Dice coefficient
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + (1 - dice)


# Dice metric calculation
def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            dice_scores.append(dice_score(outputs, masks).item())
    return running_loss / len(loader.dataset), dice_scores


def test_model(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores.append(dice_score(outputs, masks).item())
    return dice_scores


def save_losses(train_losses, val_losses, save_path):
    df = pd.DataFrame({'Epoch': list(range(1, len(train_losses) + 1)),
                       'Train Loss': train_losses,
                       'Val Loss': val_losses})
    df.to_excel(os.path.join(save_path, 'losses.xlsx'), index=False)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()


def visualize_results(model, loader, device, save_path, num_samples=5):
    model.eval()
    indices = np.random.choice(len(loader.dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle('Segmentation Results', fontsize=16)

    for i, idx in enumerate(indices):
        image, mask, img_name = loader.dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()

        axes[i, 0].imshow(image.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Input: {img_name}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'segmentation_results.png'))
    plt.close()