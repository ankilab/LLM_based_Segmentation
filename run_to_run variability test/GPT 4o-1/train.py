# unet_segmentation/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import openpyxl
import os

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()

def save_excel(data, filename):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)
    wb.save(filename)

def visualize_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(save_path)
    plt.close()

def train(model, train_loader, val_loader, device, save_dir, epochs=30, lr=1e-4):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []
    val_dice_all = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0
        dice_scores = []
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                dice = dice_score(preds, masks)
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        val_dice_all.append(dice_scores)

    save_excel([["Epoch"] + list(range(1, epochs+1)), ["Loss"] + train_loss_history], os.path.join(save_dir, "train_losses.xlsx"))
    save_excel([["Epoch"] + list(range(1, epochs+1)), ["Loss"] + val_loss_history], os.path.join(save_dir, "val_losses.xlsx"))
    save_excel(val_dice_all, os.path.join(save_dir, "validation_dice_scores.xlsx"))

    torch.save(model, os.path.join(save_dir, "model_full.pth"))
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state_dict.pth"))

    visualize_losses(train_loss_history, val_loss_history, os.path.join(save_dir, "loss_plot.png"))

def test(model, test_loader, device, save_dir):
    model.eval()
    dice_scores = []
    predictions = []
    inputs = []
    targets = []
    filenames = []

    with torch.no_grad():
        for images, masks, names in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice = dice_score(preds, masks)
            dice_scores.append(dice)
            predictions.extend(preds.cpu())
            inputs.extend(images.cpu())
            targets.extend(masks.cpu())
            filenames.extend(names)

    save_excel([dice_scores], os.path.join(save_dir, "test_dice_scores.xlsx"))

    indices = random.sample(range(len(inputs)), 5)
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(inputs[idx][0], cmap='gray')
        axes[i, 0].set_title(f"Input\n{filenames[idx]}")
        axes[i, 1].imshow(targets[idx][0], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(predictions[idx][0] > 0.5, cmap='gray')
        axes[i, 2].set_title("Prediction")
        for j in range(3):
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_predictions.png"))
    plt.close()
