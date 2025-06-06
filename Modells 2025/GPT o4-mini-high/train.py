import os
import time
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def dice_coef_per_sample(preds, targets, eps=1e-6):
    """
    Compute Dice coefficient for each sample in the batch.
    Inputs:
      preds: tensor of shape [B, 1, H, W], values 0 or 1 (after threshold)
      targets: tensor of shape [B, 1, H, W], values 0 or 1
    Returns:
      list of length B with dice value for each sample
    """
    batch_size = preds.size(0)
    dice_scores = []
    for i in range(batch_size):
        pred_flat = preds[i].view(-1)
        target_flat = targets[i].view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    return dice_scores


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for images, masks, _ in loop:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_batch_dice = []  # one dice per batch (averaged over samples in batch)

    loop = tqdm(dataloader, desc=f"Validate Epoch {epoch}", leave=False)
    with torch.no_grad():
        for images, masks, _ in loop:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Compute dice: apply sigmoid, threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            batch_dice_list = dice_coef_per_sample(preds, masks)
            batch_dice = sum(batch_dice_list) / len(batch_dice_list)
            all_batch_dice.append(batch_dice)

            loop.set_postfix(loss=loss.item(), dice=batch_dice)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, all_batch_dice


def test(model, dataloader, device):
    """
    Similar to validate but without updating weights and without cross-validation.
    Returns:
      test_loss_list: average loss per batch
      test_dice_list: dice score per batch
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_batch_dice = []
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Testing", leave=False)
    with torch.no_grad():
        for images, masks, _ in loop:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            batch_dice_list = dice_coef_per_sample(preds, masks)
            batch_dice = sum(batch_dice_list) / len(batch_dice_list)
            all_batch_dice.append(batch_dice)

            loop.set_postfix(loss=loss.item(), dice=batch_dice)

    avg_loss = running_loss / len(dataloader)
    return avg_loss, all_batch_dice


def save_losses_to_excel(loss_list, filepath):
    """
    Save a list of per-epoch losses into an Excel file:
      Row 1: epoch numbers (1..n)
      Row 2: corresponding loss values
    """
    epochs = list(range(1, len(loss_list) + 1))
    data = [epochs, loss_list]
    df = pd.DataFrame(data)
    df.to_excel(filepath, index=False, header=False)


def save_dice_scores_to_excel(dice_scores_matrix, filepath):
    """
    Save a 2D list or array: rows = epochs, columns = batches.
    Adds row and column labels (epochs and batch indices) automatically.
    """
    df = pd.DataFrame(dice_scores_matrix)
    # Label rows as Epoch 1, 2, ... and columns as Batch 1, 2, ...
    df.index = [f"Epoch {i+1}" for i in range(df.shape[0])]
    df.columns = [f"Batch {j+1}" for j in range(df.shape[1])]
    df.to_excel(filepath)


def plot_losses(train_losses, val_losses, save_path):
    """
    Plot training and validation losses over epochs and save as PNG.
    """
    plt.figure(figsize=(8, 6))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(model, dataset, device, save_path, num_samples=5):
    """
    Visualize `num_samples` random samples from `dataset`:
      For each sample: input image, ground truth mask, model prediction.
      Arrange in a figure with `num_samples` rows and 3 columns.
      Save as PNG to `save_path`.
    """
    model.eval()
    indices = list(range(len(dataset)))
    selected = random.sample(indices, k=num_samples)

    # Prepare figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.subplots_adjust(top=0.93)

    # Column titles
    col_titles = ['Input Image', 'Ground Truth', 'Prediction']
    for col_idx, title in enumerate(col_titles):
        fig.text((col_idx + 0.5) / 3, 0.97, title, ha='center', va='bottom', fontsize=14)

    with torch.no_grad():
        for row_idx, idx in enumerate(selected):
            image_tensor, mask_tensor, base_name = dataset[idx]
            img = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)  # [1,1,H,W]
            gt_mask = mask_tensor.squeeze(0).cpu().numpy()  # [H,W]

            output = model(img)
            prob = torch.sigmoid(output)
            pred_mask = (prob > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()  # [H,W]

            # Move original image to numpy for plotting
            img_np = image_tensor.squeeze(0).cpu().numpy()  # [H,W]

            for col in range(3):
                ax = axes[row_idx, col]
                ax.axis('off')
                if col == 0:
                    ax.imshow(img_np, cmap='gray')
                elif col == 1:
                    ax.imshow(gt_mask, cmap='gray')
                else:
                    ax.imshow(pred_mask, cmap='gray')
                # Filename above each subplot
                ax.set_title(base_name, fontsize=10, pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
