# unet_segmentation/train.py

import os
import time
import random
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from PIL import Image

# Dice for binary masks
def dice_coeff(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2 * intersection + eps) / (union + eps)).cpu().numpy()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    for batch_idx, (imgs, masks, _) in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            preds = model(imgs)
            loss = criterion(preds, masks)
        running_loss += loss.item() * imgs.size(0)
        dice_scores.append(dice_coeff(preds, masks))
    avg_loss = running_loss / len(dataloader.dataset)

    # save dice scores: rows=epoch, cols=batches
    df = pd.DataFrame(dice_scores).T
    fn = os.path.join(save_path, "validation_dice_scores.xlsx")
    with pd.ExcelWriter(fn, mode='a' if os.path.exists(fn) else 'w') as writer:
        df.to_excel(writer, sheet_name=f"epoch_{epoch}", index=False)
    return avg_loss

def test_model(model, dataloader, device, save_path):
    model.eval()
    dice_scores = []
    all_samples = []
    for imgs, masks, names in tqdm(dataloader, desc="Testing", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            preds = model(imgs)
        dice_scores.append(dice_coeff(preds, masks))
        all_samples.extend(zip(imgs.cpu(), masks.cpu(), (preds>0.5).cpu(), names))
    # save test dice scores in same layout
    df = pd.DataFrame(dice_scores).T
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"), index=False)
    return all_samples

def save_losses(losses, fn):
    df = pd.DataFrame([losses], index=["loss"])
    df.columns = [f"epoch_{i+1}" for i in range(len(losses))]
    df.to_excel(fn, index=False)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1,len(train_losses)+1), train_losses, label="Train")
    plt.plot(range(1,len(val_losses)+1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def visualize_predictions(samples, save_path, n=5):
    picked = random.sample(samples, n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    for i, (img, mask, pred, name) in enumerate(picked):
        axes[i,0].imshow(img.squeeze(), cmap='gray')
        axes[i,0].set_title(name)
        axes[i,0].axis('off')
        axes[i,1].imshow(mask.squeeze(), cmap='gray')
        axes[i,1].set_title("Ground Truth")
        axes[i,1].axis('off')
        axes[i,2].imshow(pred.squeeze(), cmap='gray')
        axes[i,2].set_title("Prediction")
        axes[i,2].axis('off')
    fig.suptitle("Input | Ground Truth | Prediction", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
    plt.close()
