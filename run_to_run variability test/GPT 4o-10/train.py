# unet_segmentation/train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def dice_score(pred, target, epsilon=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2. * intersection + epsilon) / (union + epsilon)).mean().item()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for img, mask, _ in pbar:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return epoch_loss / len(loader)

def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    epoch_loss = 0
    dice_scores = []
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for i, (img, mask, _) in enumerate(pbar):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            dice = dice_score(pred, mask)
            epoch_loss += loss.item()
            dice_scores.append(dice)
            pbar.set_postfix(val_loss=loss.item(), dice=dice)

    # Convert to DataFrame: each epoch is a row, each batch is a column
    df_new_row = pd.DataFrame([dice_scores])

    dice_path = os.path.join(save_path, "validation_dice_scores.xlsx")
    if os.path.exists(dice_path):
        df_existing = pd.read_excel(dice_path, header=None)
        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_excel(dice_path, header=False, index=False)
    else:
        df_new_row.to_excel(dice_path, header=False, index=False)

    return epoch_loss / len(loader)


def test(model, loader, device, save_path):
    model.eval()
    all_dice_scores = []
    predictions = []
    filenames = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for img, mask, fname in pbar:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            dice = dice_score(pred, mask)
            all_dice_scores.append(dice)
            predictions.append((img.cpu(), mask.cpu(), pred.cpu()))
            filenames.append(fname)

    # Save Dice scores in same format: each row = 1 epoch/test run
    df_new_row = pd.DataFrame([all_dice_scores])
    dice_path = os.path.join(save_path, "test_dice_scores.xlsx")
    if os.path.exists(dice_path):
        df_existing = pd.read_excel(dice_path, header=None)
        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_excel(dice_path, header=False, index=False)
    else:
        df_new_row.to_excel(dice_path, header=False, index=False)

    # Visualize 5 random predictions
    import random
    import matplotlib.pyplot as plt

    random_idx = random.sample(range(len(predictions)), min(5, len(predictions)))
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    for i, idx in enumerate(random_idx):
        img, mask, pred = predictions[idx]
        axs[i, 0].imshow(img.squeeze(), cmap='gray')
        axs[i, 0].set_title("Input Image")
        axs[i, 1].imshow(mask.squeeze(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(pred.squeeze(), cmap='gray')
        axs[i, 2].set_title("Prediction")

        for j in range(3):
            axs[i, j].axis('off')
            axs[i, j].text(0.5, -0.15, filenames[idx][0], transform=axs[i, j].transAxes,
                           fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'))
    plt.close()

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()
