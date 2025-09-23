# train.py
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for x, y, _ in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, loss_fn, device, save_path, epoch):
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for i, (x, y, _) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            val_loss += loss.item() * x.size(0)
            dice = dice_score(preds, y).item()
            dice_scores.append(dice)

    # Save Dice scores as row in Excel
    dice_excel_path = os.path.join(save_path, 'validation_dice_scores.xlsx')
    df = pd.DataFrame([dice_scores])
    with pd.ExcelWriter(dice_excel_path, mode='a' if os.path.exists(dice_excel_path) else 'w',
                        engine='openpyxl') as writer:
        df.to_excel(writer, header=False, index=False)

    return val_loss / len(dataloader.dataset)


def test(model, dataloader, device, save_path):
    model.eval()
    dice_scores = []
    all_samples = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Testing", leave=False)
        for x, y, names in loop:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            for i in range(len(x)):
                dice = dice_score(preds[i], y[i]).item()
                dice_scores.append(dice)
                all_samples.append((x[i].cpu(), y[i].cpu(), preds[i].cpu(), names[i]))

    # Save Dice scores
    df = pd.DataFrame([dice_scores])
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False, header=False)

    # Visualization of 5 random samples
    import random
    samples = random.sample(all_samples, 5)
    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    for i, (img, gt, pred, name) in enumerate(samples):
        axs[i, 0].imshow(img.squeeze(), cmap='gray')
        axs[i, 0].set_title("Input Image\n" + name)
        axs[i, 1].imshow(gt.squeeze(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(pred.squeeze(), cmap='gray')
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
