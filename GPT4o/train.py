import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


def train_fn(loader, model, optimizer, loss_fn, device, epoch):
    model.train()
    running_loss = 0
    total_samples = 0
    loop = tqdm(loader, desc=f'Epoch {epoch + 1} [Training]', leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_samples += data.size(0)

        # Update tqdm with current loss and total samples processed
        loop.set_postfix(loss=loss.item(), total_samples=total_samples, batch=batch_idx + 1)

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch + 1} completed: Total samples processed: {total_samples}, Average loss: {avg_loss:.4f}")
    return avg_loss


def validate_fn(loader, model, loss_fn, device, epoch):
    model.eval()
    running_loss = 0
    dice_scores = []
    total_samples = 0
    loop = tqdm(loader, desc=f'Epoch {epoch+1} [Validation]', leave=True)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            running_loss += loss.item()

            dice = dice_coefficient(predictions, targets).item()  # Ensure dice is a scalar
            dice_scores.append(dice)

            total_samples += data.size(0)

            # Update tqdm with current loss, dice score, and total samples processed
            loop.set_postfix(loss=loss.item(), dice=dice, total_samples=total_samples, batch=batch_idx+1)

    avg_loss = running_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"Validation completed: Total samples processed: {total_samples}, Average loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")
    return avg_loss, dice_scores  # Return dice_scores list


def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def save_losses(train_losses, val_losses, save_path):
    df_train = pd.DataFrame([train_losses], index=['Training Losses'])
    df_val = pd.DataFrame([val_losses], index=['Validation Losses'])
    df_train.to_excel(f'{save_path}/train_losses.xlsx', index=True)
    df_val.to_excel(f'{save_path}/val_losses.xlsx', index=True)

def save_dice_scores(dice_scores, save_path, filename):
    # Ensure dice_scores are already scalar values
    df = pd.DataFrame([dice_scores])  # Store them in a single row
    df.to_excel(f"{save_path}/{filename}.xlsx", index=False)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{save_path}/loss_plot.png')


def save_model(model, save_path, filename="model.pth"):
    torch.save(model.state_dict(), f"{save_path}/{filename}")
    torch.save(model, f"{save_path}/model_full.pth")


def test_fn(loader, model, device):
    model.eval()
    loop = tqdm(loader)
    dice_scores = []
    visual_data = []

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            dice = dice_coefficient(predictions, targets)
            dice_scores.append(dice)

            if len(visual_data) < 5:
                visual_data.append((data, targets, predictions))

    return dice_scores, visual_data


def visualize_predictions(samples, save_path):
    plt.figure(figsize=(10, 15))

    for i, (img_batch, target_batch, pred_batch) in enumerate(samples):
        # Select the first image from the batch (you can change this if needed)
        img = img_batch[0].squeeze(0).cpu().numpy()  # Shape (128, 128)
        target = target_batch[0].squeeze(0).cpu().numpy()  # Shape (128, 128)
        pred = pred_batch[0].squeeze(0).cpu().numpy()  # Shape (128, 128)

        plt.subplot(5, 3, 3 * i + 1)
        plt.title(f"Input Image {i + 1}")
        plt.imshow(img, cmap='gray')

        plt.subplot(5, 3, 3 * i + 2)
        plt.title(f"Ground Truth {i + 1}")
        plt.imshow(target, cmap='gray')

        plt.subplot(5, 3, 3 * i + 3)
        plt.title(f"Prediction {i + 1}")
        plt.imshow(pred, cmap='gray')

    plt.tight_layout()
    plt.savefig(f'{save_path}/predictions.png')

