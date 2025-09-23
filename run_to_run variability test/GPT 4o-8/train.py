# unet_segmentation/train.py

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os


def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersect + epsilon) / (union + epsilon)
    return dice.mean().item()


def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Training")
    for images, masks, _ in progress:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, save_path, epoch):
    model.eval()
    val_loss = 0.0
    dice_scores = []
    progress = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, masks, _ in progress:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            dice_scores.append(dice_score(outputs, masks))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    val_dice_df = pd.DataFrame([dice_scores])
    val_dice_path = os.path.join(save_path, "validation_dice_scores.xlsx")
    if os.path.exists(val_dice_path):
        existing = pd.read_excel(val_dice_path, index_col=0)
        val_dice_df = pd.concat([existing, val_dice_df], ignore_index=True)
    val_dice_df.to_excel(val_dice_path)

    return val_loss / len(loader.dataset)


def test(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    images_list = []
    outputs_list = []
    masks_list = []
    names = []
    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for images, masks, img_names in progress:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores.append(dice_score(outputs, masks))
            if len(images_list) < 5:
                images_list.extend(images.cpu())
                outputs_list.extend(outputs.cpu())
                masks_list.extend(masks.cpu())
                names.extend(img_names)

    df = pd.DataFrame([dice_scores])
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"))

    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    for i in range(5):
        axs[i, 0].imshow(images_list[i].squeeze(), cmap='gray')
        axs[i, 0].set_title(f"{names[i]}")
        axs[i, 1].imshow(masks_list[i].squeeze(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(outputs_list[i].squeeze() > 0.5, cmap='gray')
        axs[i, 2].set_title("Prediction")

        for j in range(3):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
