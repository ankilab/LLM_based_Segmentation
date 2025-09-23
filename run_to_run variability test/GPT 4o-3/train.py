# unet_segmentation/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import pandas as pd
import random
import os

def dice_score(preds, targets):
    preds = (preds > 0.5).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.cpu().numpy()

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, masks, _ in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device, save_path, epoch):
    model.eval()
    val_loss = 0.0
    all_dices = []
    batch_dices = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validating", leave=False)
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()
            batch_dice = dice_score(preds, masks)
            batch_dices.append(batch_dice)
            all_dices.extend(batch_dice)
            loop.set_postfix(loss=loss.item())
    batch_df = pd.DataFrame([np.array(d) for d in batch_dices])
    batch_df.to_excel(os.path.join(save_path, f'validation_dice_scores.xlsx'), index=False, header=False)
    return val_loss / len(dataloader)

def test_model(model, dataloader, device, save_path):
    model.eval()
    dices = []
    image_names = []
    predictions = []
    gts = []
    inputs = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Testing", leave=False)
        for imgs, masks, names in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dices.extend(dice_score(preds, masks))
            predictions.extend(preds.cpu())
            gts.extend(masks.cpu())
            inputs.extend(imgs.cpu())
            image_names.extend(names)
    df = pd.DataFrame(np.array(dices).reshape(1, -1))
    df.to_excel(os.path.join(save_path, f'test_dice_scores.xlsx'), index=False, header=False)

    # Visualization
    indices = random.sample(range(len(predictions)), 5)
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i, idx in enumerate(indices):
        axes[i][0].imshow(inputs[idx].squeeze(), cmap='gray')
        axes[i][0].set_title(f"Input\n{image_names[idx]}")
        axes[i][1].imshow(gts[idx].squeeze(), cmap='gray')
        axes[i][1].set_title("Ground Truth")
        axes[i][2].imshow(predictions[idx].squeeze() > 0.5, cmap='gray')
        axes[i][2].set_title("Prediction")
        for j in range(3):
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_predictions.png'))
