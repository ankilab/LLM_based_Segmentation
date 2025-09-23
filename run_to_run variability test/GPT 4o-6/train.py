# unet_segmentation/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)

def save_loss_excel(losses, filename):
    df = pd.DataFrame([losses], index=["Loss"])
    df.columns = [f"Epoch {i+1}" for i in range(len(losses))]
    df.to_excel(filename)

def save_dice_scores_excel(dice_scores, filename):
    df = pd.DataFrame(dice_scores)
    df.to_excel(filename, index=False)

def visualize_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, dataset, device, save_path):
    model.eval()
    indices = random.sample(range(len(dataset)), 5)
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, name = dataset[idx]
            img = img.to(device).unsqueeze(0)
            pred = model(img)
            pred = pred.squeeze().cpu().numpy()
            img = img.squeeze().cpu().numpy()
            mask = mask.squeeze().numpy()

            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 2].imshow(pred > 0.5, cmap="gray")
            axes[i, 0].set_title(f"{name} - Input")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].set_title("Prediction")

            for j in range(3):
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = imgs.unsqueeze(1), masks.unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Validate", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            imgs, masks = imgs.unsqueeze(1), masks.unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()
            dice_scores.append(dice_score(preds, masks).item())
    return val_loss / len(loader), dice_scores

def test(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Test", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            imgs, masks = imgs.unsqueeze(1), masks.unsqueeze(1)
            preds = model(imgs)
            dice_scores.append(dice_score(preds, masks).item())
    return dice_scores
