# unet_segmentation/train.py

import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    pred, target: each [B,1,H,W], binary {0,1}
    returns mean dice per batch
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return ((2 * intersection + eps) / (pred.sum(1) + target.sum(1) + eps)).mean().item()

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    avg_loss = running_loss / n_batches
    return avg_loss

def validate_epoch(model, loader, criterion, device, epoch, save_path):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    n_batches = len(loader)
    pbar = tqdm(loader, desc=f"Val Epoch {epoch+1}", leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        dice_scores.append(dice_coeff(preds, masks))
        pbar.set_postfix({'val_loss': loss.item(), 'dice': dice_scores[-1]})

    avg_loss = running_loss / n_batches

    # --- FIXED: ensure DataFrame has columns before inserting first row ---
    dice_file = os.path.join(save_path, "validation_dice_scores.xlsx")
    batch_cols = [f"Batch_{i+1}" for i in range(n_batches)]
    if os.path.exists(dice_file):
        df = pd.read_excel(dice_file, index_col=0)
    else:
        # initialize empty DF with the correct batch columns
        df = pd.DataFrame(columns=batch_cols)

    # if loaded df has fewer/more columns (e.g. batch count changed), reindex:
    df = df.reindex(columns=batch_cols)

    # now assign the new row
    df.loc[f"Epoch_{epoch+1}"] = dice_scores
    df.to_excel(dice_file)

    return avg_loss


def test_model(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    pbar = tqdm(loader, desc="Testing", leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        preds = (outputs > 0.5).float()
        dice_scores.append(dice_coeff(preds, masks))
    # save dice scores
    dice_file = os.path.join(save_path, "test_dice_scores.xlsx")
    df = pd.DataFrame([dice_scores], index=["Dice"]).T
    df.to_excel(dice_file, header=False)
    return

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(save_path, "loss_curve.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
