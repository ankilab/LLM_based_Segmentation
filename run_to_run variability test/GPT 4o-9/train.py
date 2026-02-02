# unet_segmentation/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def save_losses(losses, filename, save_path):
    df = pd.DataFrame([list(range(1, len(losses)+1)), losses])
    df.to_excel(os.path.join(save_path, filename), header=False, index=False)


def save_dice_scores(dice_scores, filename, save_path):
    df = pd.DataFrame(dice_scores)
    df.to_excel(os.path.join(save_path, filename), index=False)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()


def visualize_predictions(model, dataloader, save_path, device):
    model.eval()
    samples = random.sample(range(len(dataloader.dataset)), 5)
    inputs, masks, names = zip(*[dataloader.dataset[i] for i in samples])

    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    axs[0][0].set_title("Input Image")
    axs[0][1].set_title("Ground Truth")
    axs[0][2].set_title("Prediction")

    with torch.no_grad():
        for i in range(5):
            image = inputs[i].unsqueeze(0).to(device)
            output = model(image).squeeze().cpu().numpy()
            axs[i][0].imshow(inputs[i].squeeze(), cmap="gray")
            axs[i][0].set_ylabel(names[i], fontsize=8)
            axs[i][1].imshow(masks[i].squeeze(), cmap="gray")
            axs[i][2].imshow(output > 0.5, cmap="gray")

    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "prediction_visualization.png"))
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for inputs, masks, _ in loop:
        inputs, masks = inputs.to(device), masks.to(device)
        preds = model(inputs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, save_path, epoch):
    model.eval()
    val_loss = 0
    dice_scores = []

    loop = tqdm(loader, desc="Validation", leave=False)
    for inputs, masks, _ in loop:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, masks)
            val_loss += loss.item() * inputs.size(0)
            batch_dice = dice_score(preds, masks)
            dice_scores.append(batch_dice)

    return val_loss / len(loader.dataset), dice_scores


def test(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    loop = tqdm(loader, desc="Testing")
    for inputs, masks, _ in loop:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.no_grad():
            preds = model(inputs)
            batch_dice = dice_score(preds, masks)
            dice_scores.append(batch_dice)
    return dice_scores
