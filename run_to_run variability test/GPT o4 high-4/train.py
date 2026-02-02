# train.py

import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    Compute per-sample Dice coefficient.
    pred: sigmoid’d output, shape [B,1,H,W], values in [0,1]
    target: binary mask, same shape
    """
    pred_bin = (pred > 0.5).float()
    num = 2 * (pred_bin * target).sum(dim=[1,2,3])
    den = pred_bin.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3]) + eps
    return num / den

def train_epoch(model, loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_per_batch = []
    pbar = tqdm(loader, desc='Validate', leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            logits = model(imgs)
            loss = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits)
            dice = dice_coef(probs, masks)
            batch_dice = float(dice.mean().item())
            dice_per_batch.append(batch_dice)
        pbar.set_postfix(loss=loss.item(), dice=batch_dice)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(dice_per_batch)

def test_model(model, loader: DataLoader, device):
    model.eval()
    dice_per_batch = []
    pbar = tqdm(loader, desc='Test', leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            dice = dice_coef(probs, masks)
            batch_dice = float(dice.mean().item())
            dice_per_batch.append(batch_dice)
        pbar.set_postfix(dice=batch_dice)
    return np.array(dice_per_batch)

def save_losses_to_excel(losses: list, epochs: list, path: str, filename: str):
    df = pd.DataFrame([losses], index=['loss'], columns=[f'Epoch_{e}' for e in epochs])
    df.to_excel(f'{path}/{filename}', index=True)

def save_dice_excel(dice_matrix: np.ndarray, path: str, filename: str):
    if dice_matrix.ndim == 1:
        dice_matrix = dice_matrix[np.newaxis, :]
    rows = ['Epoch_'+str(i+1) for i in range(dice_matrix.shape[0])] \
           if dice_matrix.shape[0] > 1 else ['Test']
    cols = [f'Batch_{j+1}' for j in range(dice_matrix.shape[1])]
    df = pd.DataFrame(dice_matrix, index=rows, columns=cols)
    df.to_excel(f'{path}/{filename}', index=True)

def plot_losses(train_losses, val_losses, path, filename='loss_curve.png'):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/{filename}')
    plt.close()
