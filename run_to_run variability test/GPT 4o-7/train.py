# unet_segmentation/train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import random
import openpyxl
from torch.utils.data import Subset
from torchvision.utils import make_grid

def dice_score(preds, targets):
    preds = (preds > 0.5).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2. * intersection + smooth) / (union + smooth)).cpu().numpy()

def save_excel(data, path):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)
    wb.save(path)

def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, masks, _ in loop:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device, save_path, epoch):
    model.eval()
    val_loss = 0
    dice_batches = []
    loop = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, masks, _ in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            dice = dice_score(outputs, masks)
            dice_batches.append(dice.tolist())
    save_excel(dice_batches, os.path.join(save_path, f"validation_dice_scores_epoch_{epoch}.xlsx"))
    return val_loss / len(loader)

def test(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    loop = tqdm(loader, desc="Testing", leave=False)
    all_samples = []
    with torch.no_grad():
        for images, masks, names in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.tolist())
            for i in range(len(images)):
                all_samples.append((images[i].cpu(), masks[i].cpu(), outputs[i].cpu(), names[i]))
    save_excel(dice_scores, os.path.join(save_path, "test_dice_scores.xlsx"))

    # Visualize 5 random samples
    samples = random.sample(all_samples, 5)
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    for i, (img, gt, pred, name) in enumerate(samples):
        axs[i, 0].imshow(img.squeeze(), cmap='gray')
        axs[i, 1].imshow(gt.squeeze(), cmap='gray')
        axs[i, 2].imshow(pred.squeeze(), cmap='gray')
        axs[i, 0].set_title(f"Input: {name}")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
    plt.close()

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.close()
