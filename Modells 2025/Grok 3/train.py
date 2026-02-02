import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import jaccard_score

def dice_score(pred, target, smooth=1):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks, _ in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_model(model, val_loader, criterion, device, epoch, save_path):
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            batch_dice = dice_score(preds, masks)
            dice_scores.append(batch_dice.item())
    # Save dice scores for this epoch
    dice_df = pd.DataFrame([dice_scores])
    dice_file = os.path.join(save_path, "validation_dice_scores.xlsx")
    if os.path.exists(dice_file):
        existing_df = pd.read_excel(dice_file)
        dice_df = pd.concat([existing_df, dice_df], ignore_index=True)
    dice_df.to_excel(dice_file, index=False)
    return val_loss / len(val_loader), np.mean(dice_scores)

def test_model(model, test_loader, device, save_path):
    model.eval()
    dice_scores = []
    predictions, ground_truths, input_images, filenames = [], [], [], []
    with torch.no_grad():
        for images, masks, names in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            batch_dice = dice_score(preds, masks)
            dice_scores.append(batch_dice.item())
            predictions.append(preds.cpu())
            ground_truths.append(masks.cpu())
            input_images.append(images.cpu())
            filenames.extend(names)
    # Save dice scores
    dice_df = pd.DataFrame([dice_scores])
    dice_df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"), index=False)
    return predictions, ground_truths, input_images, filenames

def save_losses(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))
    pd.DataFrame([epochs, train_losses]).T.to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False, header=["Epoch", "Loss"])
    pd.DataFrame([epochs, val_losses]).T.to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False, header=["Epoch", "Loss"])

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.close()

def visualize_predictions(predictions, ground_truths, input_images, filenames, save_path):
    num_samples = 5
    indices = np.random.choice(len(filenames), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(input_images[idx // input_images[0].shape[0]][idx % input_images[0].shape[0], 0].numpy(), cmap='gray')
        axes[i, 0].set_title(f"Input: {filenames[idx]}")
        axes[i, 1].imshow(ground_truths[idx // ground_truths[0].shape[0]][idx % ground_truths[0].shape[0], 0].numpy(), cmap='gray')
        axes[i, 1].set_title(f"Ground Truth: {filenames[idx]}")
        axes[i, 2].imshow(predictions[idx // predictions[0].shape[0]][idx % predictions[0].shape[0], 0].numpy(), cmap='gray')
        axes[i, 2].set_title(f"Prediction: {filenames[idx]}")
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions.png"))
    plt.close()