# unet_segmentation/train.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import jaccard_score


def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, masks, _ in loop:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(model, loader, criterion, device, save_path, epoch):
    model.eval()
    epoch_loss = 0
    dice_scores = []
    dice_per_batch = []

    loop = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, masks, _ in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            epoch_loss += loss.item()
            dice = dice_score(preds, masks)
            dice_scores.append(dice)
            dice_per_batch.append(dice)

    if epoch == 0:
        df = pd.DataFrame(columns=[f"Batch_{i}" for i in range(len(dice_per_batch))])
    else:
        df = pd.read_excel(os.path.join(save_path, "validation_dice_scores.xlsx"), index_col=0)

    df.loc[f"Epoch_{epoch}"] = dice_per_batch
    df.to_excel(os.path.join(save_path, "validation_dice_scores.xlsx"))

    return epoch_loss / len(loader)


def test(model, loader, device, save_path):
    model.eval()
    dice_per_batch = []
    predictions = []

    loop = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for images, masks, names in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice = dice_score(preds, masks)
            dice_per_batch.append(dice)
            predictions.append((images.cpu(), masks.cpu(), preds.cpu(), names))

    df = pd.DataFrame(columns=[f"Batch_{i}" for i in range(len(dice_per_batch))])
    df.loc["Dice"] = dice_per_batch
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"))

    return predictions


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()


def visualize_predictions(predictions, save_path):
    import matplotlib.pyplot as plt

    samples = predictions[:5]
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    titles = ["Input Image", "Ground Truth", "Prediction"]

    for i, (img, mask, pred, name) in enumerate(samples):
        axs[i, 0].imshow(img[0][0], cmap='gray')
        axs[i, 1].imshow(mask[0][0], cmap='gray')
        axs[i, 2].imshow(pred[0][0] > 0.5, cmap='gray')
        for j in range(3):
            axs[i, j].axis("off")
            axs[i, j].set_title(titles[j])
            axs[i, j].text(0.5, -0.15, name[0], size=10, ha="center", transform=axs[i, j].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_visualization.png"))
    plt.close()
