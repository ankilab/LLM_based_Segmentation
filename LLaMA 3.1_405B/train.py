import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def calculate_dice_score(outputs, masks):
    outputs = (outputs > 0.5).float()
    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum()
    return 2 * intersection / union

def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    with tqdm(loader, desc=f'Epoch {epoch+1}') as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Resize masks to match output size
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def validate(model, device, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    dice_scores = []
    with tqdm(loader, desc=f'Validation Epoch {epoch+1}') as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                outputs = model(images)
                # Resize masks to match output size
                masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                # Calculate Dice score
                outputs = (outputs > 0.5).float()
                dice_score = 2 * (outputs * masks).sum() / (outputs.sum() + masks.sum())
                dice_scores.append(dice_score.item())
    return total_loss / len(loader), np.mean(dice_scores)

def test(model, device, loader):
    model.eval()
    dice_scores = []
    with tqdm(loader, desc='Testing') as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                outputs = model(images)
                # Resize masks to match output size
                masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
                outputs = (outputs > 0.5).float()
                dice_score = calculate_dice_score(outputs, masks)
                dice_scores.append(dice_score.item())
    return np.mean(dice_scores)

def save_losses(train_losses, val_losses, save_path):
    train_df = pd.DataFrame(train_losses)
    val_df = pd.DataFrame(val_losses)
    train_df.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)
    val_df.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)

# def visualize_losses(train_losses, val_losses, save_path):
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.legend()
#     plt.savefig(os.path.join(save_path, 'losses.png'))

def visualize_losses(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'orange', label='Validation loss')
    plt.title('Training and Validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))
    plt.close()

# def visualize_predictions(model, device, loader, save_path):
#     model.eval()
#     with torch.no_grad():
#         images, masks = next(iter(loader))
#         images, masks = images.to(device), masks.to(device)
#         outputs = model(images)
#         # Resize masks to match output size
#         masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
#         outputs = (outputs > 0.5).float()
#         # Save predictions as images
#         for i in range(images.shape[0]):
#             image = images[i].cpu().numpy().squeeze()
#             mask = masks[i].cpu().numpy().squeeze()
#             output = outputs[i].cpu().numpy().squeeze()
#             plt.figure(figsize=(12, 4))
#             plt.subplot(1, 3, 1)
#             plt.imshow(image, cmap='gray')
#             plt.title('Image')
#             plt.subplot(1, 3, 2)
#             plt.imshow(mask, cmap='gray')
#             plt.title('Mask')
#             plt.subplot(1, 3, 3)
#             plt.imshow(output, cmap='gray')
#             plt.title('Prediction')
#             plt.savefig(os.path.join(save_path, f'prediction_{i}.png'))
#             plt.close()

def visualize_predictions(model, dataloader, device, save_path):
    model.eval()
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    axs = axs.ravel()

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i == 5:
                break
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # Resize masks to match output size
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            outputs = (outputs > 0.5).float()
            axs[i * 3].imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 1].imshow(masks[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 2].imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')

    plt.savefig(f'{save_path}/predictions.png')
    plt.close()