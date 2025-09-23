# train.py
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset import GrayscaleSegmentationDataset
from model import UNet

def dice_coef(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(tqdm(loader, desc="Val", leave=False)):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            running_loss += loss.item()
            batch_dice = dice_coef(preds>0.5, masks).item()
            dice_scores.append(batch_dice)
    avg_loss = running_loss / len(loader)
    # save dice scores per batch in Excel sheet, one row per epoch
    df = pd.DataFrame([dice_scores])
    df.index = [epoch]
    df.to_excel(f"{save_path}/validation_dice_scores.xlsx", header=False)
    return avg_loss

def test_epoch(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Test", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice_scores.append(dice_coef(preds>0.5, masks).item())
    # save all test dice scores
    df = pd.DataFrame([dice_scores])
    df.to_excel(f"{save_path}/test_dice_scores.xlsx", header=False)
    return dice_scores

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path}/loss_curve.png")
    plt.close()

def plot_predictions(model, dataset, device, save_path, n_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), n_samples)
    fig, axs = plt.subplots(n_samples, 3, figsize=(8, 4*n_samples))
    for i, idx in enumerate(indices):
        img, mask, name = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze().detach().numpy() > 0.5
        axs[i,0].imshow(img.squeeze(), cmap="gray"); axs[i,0].set_title(f"Input\n{name}")
        axs[i,1].imshow(mask.squeeze(), cmap="gray"); axs[i,1].set_title("GT")
        axs[i,2].imshow(pred, cmap="gray");       axs[i,2].set_title("Pred")
        for ax in axs[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_path}/predictions.png")
    plt.close()
