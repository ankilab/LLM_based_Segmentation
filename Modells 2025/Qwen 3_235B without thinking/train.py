# unet_segmentation/train.py

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import random
import numpy as np

BCE_LOSS = nn.BCELoss()

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    all_dice_scores = []

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = BCE_LOSS(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_dice = dice_score(outputs, masks).item()
        all_dice_scores.append(batch_dice)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, all_dice_scores

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_dice_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = BCE_LOSS(outputs, masks)
            running_loss += loss.item()
            batch_dice = dice_score(outputs, masks).item()
            all_dice_scores.append(batch_dice)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, all_dice_scores

def test(model, dataloader, device, save_path="test_results.xlsx"):
    model.eval()
    all_dice_scores = []
    selected_samples = []

    indices = list(range(len(dataloader.dataset)))
    sampled_indices = random.sample(indices, min(5, len(indices)))

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_dice = [dice_score(o, m).item() for o, m in zip(outputs, masks)]
            all_dice_scores.extend(batch_dice)

            if i in sampled_indices:
                selected_samples.append((images.cpu(), masks.cpu(), outputs.cpu()))

    # Save Dice scores
    df = pd.DataFrame([all_dice_scores])
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"), index=False)

    # Plot samples
    plot_sample_predictions(selected_samples, save_path)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def plot_sample_predictions(samples, save_path):
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, len(samples)*3))
    if len(samples) == 1:
        axes = [axes]

    for i, (imgs, masks, preds) in enumerate(samples):
        for j in range(min(1, imgs.shape[0])):
            img = imgs[j].squeeze().numpy()
            mask = masks[j].squeeze().numpy()
            pred = preds[j].squeeze().numpy()

            axes[i][0].imshow(img, cmap='gray')
            axes[i][0].set_title("Input Image")

            axes[i][1].imshow(mask, cmap='gray')
            axes[i][1].set_title("Ground Truth")

            axes[i][2].imshow(pred, cmap='gray')
            axes[i][2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sample_predictions.png"))
    plt.close()