import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred) > 0.5
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    return (2. * intersection + smooth) / (union + smooth)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in tqdm(dataloader, desc='Training'):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc='Validating'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            dice_scores.extend(dice_coefficient(outputs, masks).cpu().numpy())
    return running_loss / len(dataloader.dataset), dice_scores

def test(model, dataloader, device, save_path):
    model.eval()
    dice_scores = []
    all_images = []
    all_masks = []
    all_preds = []
    all_names = []

    with torch.no_grad():
        for images, masks, names in tqdm(dataloader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            #dice_scores.extend(dice_coefficient(outputs, masks).cpu().numpy())
            # compute per-sample dice for this batch, then append the batch-mean
            batch_dice = dice_coefficient(outputs, masks)
            batch_mean = batch_dice.mean().item()  # tensor of shape (batch,)
            dice_scores.append(batch_mean)  # single float

            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(preds.cpu())
            all_names.extend(names)

    # Concatenate
    all_images = torch.cat(all_images)
    all_masks = torch.cat(all_masks)
    all_preds = torch.cat(all_preds)

    # Visualize predictions
    visualize_predictions(all_images, all_masks, all_preds, all_names, save_path)

    return dice_scores

def visualize_predictions(images, masks, preds, names, save_path):
    idxs = np.random.choice(len(images), size=5, replace=False)
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for i, idx in enumerate(idxs):
        axes[i, 0].imshow(images[idx].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Input: {names[idx]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds[idx].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_predictions.png'))
    plt.close()

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()