import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score

# Dice coefficient for binary masks
def dice_coeff(pred, target, eps=1e-6):
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1).float()
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_dice = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validate", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks.float())
            running_loss += loss.item() * imgs.size(0)
            # batch dice
            batch_dice = [dice_coeff(p, m) for p, m in zip(preds, masks)]
            all_dice.append(batch_dice)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, all_dice

def test(model, loader, device):
    model.eval()
    all_dice = []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Test", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            batch_dice = [dice_coeff(p, m) for p, m in zip(preds, masks)]
            all_dice.append(batch_dice)
    return all_dice

def save_losses_excel(losses, path, filename):
    """
    losses: list of floats
    saves two-row Excel: first row epochs, second row losses
    """
    df = pd.DataFrame([losses], index=["loss"], columns=[f"epoch_{i+1}" for i in range(len(losses))])
    df.to_excel(os.path.join(path, filename), index=True)

def save_dice_excel(dice_lists, path, filename):
    """
    dice_lists: list of lists: one list per epoch (or test), values per batch
    rows = epochs (or single row per test?), cols = batches
    """
    df = pd.DataFrame.from_records(dice_lists)
    df.index = [f"epoch_{i+1}" for i in range(len(dice_lists))]
    df.to_excel(os.path.join(path, filename), index=True)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def visualize_predictions(model, dataset, device, save_path, n_samples=5, seed=42):
    random.seed(seed)
    idxs = random.sample(range(len(dataset)), n_samples)
    imgs = []
    gts = []
    preds = []
    names = []
    model.eval()
    with torch.no_grad():
        for idx in idxs:
            img, mask = dataset[idx]
            name = os.path.basename(dataset.image_paths[idx])
            input_tensor = img.unsqueeze(0).to(device)
            pred = model(input_tensor).squeeze().cpu()
            imgs.append(img.squeeze().cpu().numpy())
            gts.append(mask.squeeze().cpu().numpy())
            preds.append((pred > 0.5).numpy())
            names.append(name)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3*n_samples))
    titles = ["Input Image", "Ground Truth", "Prediction"]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t)
    for i in range(n_samples):
        for j, data in enumerate([imgs[i], gts[i], preds[i]]):
            ax = axes[i, j]
            ax.imshow(data, cmap="gray")
            ax.axis("off")
            ax.set_xlabel(names[i] if j==0 else "")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
    plt.close()
