# unet_segmentation/train.py

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score  # can approximate Dice
from model import UNet

def dice_coeff(pred, target, eps=1e-6):
    """Batch-wise Dice coeff for binary masks."""
    pred = (pred > 0.5).float().view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2*intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    all_dice = []
    batch_dices = []
    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(tqdm(loader, desc='Val', leave=False)):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
            dice = dice_coeff(outputs, masks).cpu().numpy()
            batch_dices.append(dice)
    avg_loss = running_loss / len(loader.dataset)
    # save batch-wise dice for this epoch
    df = pd.DataFrame(batch_dices).T
    df.to_excel(os.path.join(save_path, f'validation_dice_scores.xlsx'), index=False)
    return avg_loss

def test_epoch(model, loader, device, save_path):
    model.eval()
    batch_dices = []
    names = []
    with torch.no_grad():
        for imgs, masks, batch_names in tqdm(loader, desc='Test', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            dice = dice_coeff(outputs, masks).cpu().numpy()
            batch_dices.append(dice)
            names.extend(batch_names)
    df = pd.DataFrame(batch_dices).T
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)
    return

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def visualize_predictions(model, loader, device, save_path, num_samples=5):
    import random
    model.eval()
    imgs_list, masks_list, preds_list, names = [], [], [], []
    with torch.no_grad():
        for imgs, masks, batch_names in loader:
            imgs_list.append(imgs)
            masks_list.append(masks)
            preds = model(imgs.to(device)).cpu()
            preds_list.append(preds)
            names.extend(batch_names)
    imgs_all = torch.cat(imgs_list)
    masks_all = torch.cat(masks_list)
    preds_all = torch.cat(preds_list)
    idxs = random.sample(range(len(names)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
    for i, idx in enumerate(idxs):
        axes[i,0].imshow(imgs_all[idx][0], cmap='gray')
        axes[i,0].set_title(f"{names[idx]}")
        axes[i,1].imshow(masks_all[idx][0], cmap='gray')
        axes[i,2].imshow(preds_all[idx][0]>0.5, cmap='gray')
        if i==0:
            axes[i,0].set_xlabel('Input')
            axes[i,1].set_xlabel('Ground Truth')
            axes[i,2].set_xlabel('Prediction')
        for j in range(3):
            axes[i,j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'))
    plt.close()
