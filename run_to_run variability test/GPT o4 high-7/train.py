import time
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dice_coeff_batch(preds, targets, smooth=1e-6):
    """
    preds: (B,1,H,W) raw logits
    targets: (B,1,H,W) binary masks {0,1}
    returns: list of dice per sample
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    dice_scores = []
    for p, t in zip(preds, targets):
        p_flat = p.view(-1)
        t_flat = t.view(-1)
        intersection = (p_flat * t_flat).sum()
        dice = (2.*intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)
        dice_scores.append(dice.item())
    return dice_scores

def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss

def validate_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    per_batch_means = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc='Val', leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = F.binary_cross_entropy_with_logits(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # compute mean dice for this batch
            dices = dice_coeff_batch(outputs, masks)  # list of dice per sample
            per_batch_means.append(sum(dices) / len(dices))

    avg_loss = running_loss / len(loader.dataset)
    # return both avg loss and list of per-batch mean dice
    return avg_loss, per_batch_means

def test_epoch(model, loader, device):
    model.eval()
    per_batch_means = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc='Test', leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices = dice_coeff_batch(outputs, masks)
            per_batch_means.append(sum(dices) / len(dices))
    return per_batch_means

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), bbox_inches='tight')
    plt.close()
