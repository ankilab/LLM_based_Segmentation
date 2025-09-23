# unet_segmentation/train.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)
    return dice

def save_losses(train_losses, val_losses, save_path):
    train_df = pd.DataFrame([train_losses], index=["Train"], columns=[f"Epoch {i+1}" for i in range(len(train_losses))])
    val_df = pd.DataFrame([val_losses], index=["Val"], columns=[f"Epoch {i+1}" for i in range(len(val_losses))])
    train_df.to_excel(os.path.join(save_path, "train_losses.xlsx"))
    val_df.to_excel(os.path.join(save_path, "val_losses.xlsx"))

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def save_dice_scores(dice_matrix, save_path, file_name):
    df = pd.DataFrame(dice_matrix)
    df.to_excel(os.path.join(save_path, f"{file_name}.xlsx"), index=False)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    train_losses, val_losses = [], []
    validation_dice_scores = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Train", leave=False)
        for images, masks, _ in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Val", leave=False)
            for images, masks, _ in loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice = dice_score(outputs, masks)
                val_dice_scores.append(val_dice.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        validation_dice_scores.append(np.mean(val_dice_scores, axis=0))

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    torch.save(model, os.path.join(save_path, "unet_full_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_state_dict.pth"))

    save_losses(train_losses, val_losses, save_path)
    plot_losses(train_losses, val_losses, save_path)
    save_dice_scores(validation_dice_scores, save_path, "validation_dice_scores")

def test_model(model, test_loader, device, save_path):
    model.eval()
    test_dice_scores = []
    predictions = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, masks, names in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_score(outputs, masks)
            test_dice_scores.append(dice.cpu().numpy())
            predictions.append((images.cpu(), masks.cpu(), outputs.cpu(), names))

    save_dice_scores(test_dice_scores, save_path, "test_dice_scores")
    visualize_predictions(predictions, save_path)

def visualize_predictions(predictions, save_path):
    samples = random.sample(predictions, 5)
    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    for i, (img, mask, pred, name) in enumerate(samples):
        axes[i, 0].imshow(img[0][0], cmap='gray')
        axes[i, 0].set_title(f"Input\n{name[0]}")
        axes[i, 1].imshow(mask[0][0], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred[0][0] > 0.5, cmap='gray')
        axes[i, 2].set_title("Prediction")
        for j in range(3):
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions.png"))
    plt.close()
