import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import time
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = nn.BCELoss()
#         self.dice = DiceLoss()
#
#     def forward(self, pred, target):
#         return self.bce(pred, target) + self.dice(pred, target)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Ensure inputs are properly shaped
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return self.bce(pred, target) + self.dice(pred, target)


def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            running_loss += loss.item()

            # Calculate Dice score
            preds = (outputs > 0.5).float()
            dice = dice_coeff(preds, masks.unsqueeze(1))
            dice_scores.append(dice.item())

    return running_loss / len(loader), dice_scores


def dice_coeff(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def save_losses(train_losses, val_losses, save_path):
    df_train = pd.DataFrame({'Epoch': range(1, len(train_losses) + 1), 'Train Loss': train_losses})
    df_val = pd.DataFrame({'Epoch': range(1, len(val_losses) + 1), 'Validation Loss': val_losses})

    df_train.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)
    df_val.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)


def save_dice_scores(dice_scores, filename, save_path):
    df = pd.DataFrame(dice_scores)
    df.to_excel(os.path.join(save_path, filename), index=False)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()


def save_model(model, save_path):
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict.pth'))
    torch.save(model, os.path.join(save_path, 'model.pth'))


def visualize_results(model, loader, device, save_path, num_samples=5):
    model.eval()
    images, masks, names = next(iter(loader))
    indices = torch.randperm(len(images))[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle('Segmentation Results', fontsize=16)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image = images[idx].unsqueeze(0).to(device)
            mask = masks[idx]
            name = names[idx]

            pred = model(image).cpu().squeeze()
            pred = (pred > 0.5).float()

            # Plot input image
            axes[i, 0].imshow(image.cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Input: {name}')
            axes[i, 0].axis('off')

            # Plot ground truth
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Plot prediction
            axes[i, 2].imshow(pred.squeeze(), cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sample_results.png'))
    plt.close()