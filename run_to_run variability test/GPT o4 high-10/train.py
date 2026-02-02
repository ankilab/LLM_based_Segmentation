# train.py

import os
import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score  # but we'll implement Dice ourselves

def dice_coef(pred, target, smooth=1e-6):
    """Compute per-batch Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    batch_dices = []
    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(tqdm(loader, desc='Validate', leave=False)):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
            d = dice_coef(outputs, masks).item()
            batch_dices.append(d)
    epoch_loss = running_loss / len(loader.dataset)

    # save dice scores per batch for this epoch
    df = pd.DataFrame([batch_dices])
    df.index = [f"epoch_{epoch}"]
    path = os.path.join(save_path, 'validation_dice_scores.xlsx')
    if os.path.exists(path):
        existing = pd.read_excel(path, index_col=0)
        combined = pd.concat([existing, df], axis=0)
    else:
        combined = df
    combined.to_excel(path)
    return epoch_loss

def test_model(model, loader, device, save_path):
    model.eval()
    batch_dices = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc='Test', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            batch_dices.append(dice_coef(outputs, masks).item())
    df = pd.DataFrame([batch_dices])
    df.index = ['test']
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'))
    return batch_dices

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
