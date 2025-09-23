# unet_segmentation/train.py
import os
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def dice_coeff(pred, target, eps=1e-6):
    # pred, target: tensors of shape B×1×H×W, binary {0,1}
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat  = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    return ((2*intersection + eps) / (pred_flat.sum(1) + tgt_flat.sum(1) + eps)).mean().item()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc=" Train ", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device, save_path, epoch_idx, all_val_dice):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc=" Val   ", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            loss = criterion(preds, masks)
            running_loss += loss.item() * imgs.size(0)
            bin_preds = (preds > 0.5).float()
            batch_dice = dice_coeff(bin_preds, masks)
            dice_scores.append(batch_dice)
    avg_loss = running_loss / len(loader.dataset)
    all_val_dice.append(dice_scores)
    # after epoch, save dice scores so far
    df = pd.DataFrame(all_val_dice,
                      index=[f"Epoch_{i+1}" for i in range(len(all_val_dice))])
    df.to_excel(os.path.join(save_path, "validation_dice_scores.xlsx"))
    return avg_loss

def test_epoch(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    all_preds = []
    img_names = []
    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc=" Test  ", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            bin_preds = (preds > 0.5).float()
            dice_scores.append(dice_coeff(bin_preds, masks))
            all_preds.extend(bin_preds.cpu())
            img_names.extend(names)
    # save dice scores
    df = pd.DataFrame([dice_scores], index=["Test"])
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"))
    return all_preds, img_names

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def visualize_predictions(all_preds, test_dataset, img_names, save_path):
    import matplotlib.pyplot as plt
    # randomly pick 5
    idxs = random.sample(range(len(all_preds)), 5)
    fig, axes = plt.subplots(5, 3, figsize=(9,15))
    for i, ax_row in enumerate(axes):
        img, mask, _ = test_dataset[idxs[i]]
        pred = all_preds[idxs[i]][0]
        ax_row[0].imshow(img.squeeze(), cmap="gray");    ax_row[0].set_title("Input")
        ax_row[1].imshow(mask.squeeze(), cmap="gray");   ax_row[1].set_title("Ground Truth")
        ax_row[2].imshow(pred, cmap="gray");             ax_row[2].set_title("Prediction")
        for ax in ax_row:
            ax.axis("off")
        # file name above row
        fig.suptitle("Input Image      Ground Truth      Prediction", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
    plt.close()
