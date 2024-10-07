import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time

def dice_coef(preds, targets, smooth=1.):
    preds = (preds > 0.5).float()  # Convert probabilities to binary predictions
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

    for i, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")

    with torch.no_grad():
        for i, (images, masks) in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            dice = dice_coef(outputs, masks)
            dice_scores.append(dice)

            pbar.set_postfix({'loss': loss.item(), 'dice': dice})

    epoch_loss = running_loss / len(val_loader)
    return epoch_loss, dice_scores


def test(model, test_loader, device):
    model.eval()
    dice_scores = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")

    with torch.no_grad():
        for i, (images, masks) in pbar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            dice = dice_coef(outputs, masks)
            dice_scores.append(dice)
            pbar.set_postfix({'dice': dice})
    return dice_scores


def visualize_predictions(model, test_loader, device, save_path, num_samples=5):
    model.eval()
    indices = np.random.choice(len(test_loader.dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = test_loader.dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            prediction = (prediction > 0.5).float() #binary

            image = image.cpu().numpy().squeeze()
            mask = mask.cpu().numpy().squeeze()
            prediction = prediction.cpu().numpy().squeeze()

            file_id = test_loader.dataset.image_files[idx].split('.')[0]

            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(f"{file_id}\nInput Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f"{file_id}\nGround Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(prediction, cmap='gray')
            axes[i, 2].set_title(f"{file_id}\nPrediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions_visualization.png"))



def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
