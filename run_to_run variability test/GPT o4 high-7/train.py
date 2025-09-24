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

def validate_epoch(model, loader, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    all_dices = []
    batch_idx = []
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(tqdm(loader, desc='Val', leave=False)):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss = F.binary_cross_entropy_with_logits(outputs, masks)
            running_loss += loss.item() * images.size(0)
            dices = dice_coeff_batch(outputs, masks)
            all_dices.append(dices)
            batch_idx.append(i)
    avg_loss = running_loss / len(loader.dataset)
    # save dice for this epoch
    df = pd.DataFrame(all_dices).T  # rows=epoch? we'll stack later
    dice_file = os.path.join(save_path, 'validation_dice_scores.xlsx')
    with pd.ExcelWriter(dice_file, mode='a' if os.path.exists(dice_file) else 'w') as writer:
        df.to_excel(writer, sheet_name=f'epoch_{epoch}', index=False)
    return avg_loss

def test_epoch(model, loader, device, save_path):
    model.eval()
    all_dices = []
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(tqdm(loader, desc='Test', leave=False)):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            dices = dice_coeff_batch(outputs, masks)
            all_dices.append(dices)
    df = pd.DataFrame(all_dices).T
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)
    return

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
