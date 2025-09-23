# unet_segmentation/train.py

import time
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def dice_coeff(pred, target, eps=1e-6):
    # pred, target: (B, 1, H, W)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    all_dice = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validate", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            running_loss += loss.item() * images.size(0)
            all_dice.append(dice_coeff(preds, masks).cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    # save dice per batch
    df = pd.DataFrame(all_dice)
    df.to_excel(os.path.join(save_path, f'validation_dice_scores_epoch{epoch}.xlsx'),
                index=False)
    return avg_loss


def test_model(model, loader, device, save_path):
    model.eval()
    all_dice = []
    samples = []
    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Test", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            all_dice.append(dice_coeff(preds, masks).cpu().numpy())
            # collect for visualization
            for i in range(images.size(0)):
                samples.append((images[i].cpu(), masks[i].cpu(), preds[i].cpu(), names[i]))
    # save test dice
    df = pd.DataFrame(all_dice)
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)
    return samples


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()
