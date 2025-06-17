import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    with tqdm(loader, desc='Training') as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)

def validate(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    dice_scores = []
    with torch.no_grad():
        with tqdm(loader, desc='Validation') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                dice_score = dice_coefficient(output, target)
                dice_scores.append(dice_score.item())
                pbar.set_postfix({'loss': loss.item(), 'dice': dice_score.item()})
    return total_loss / len(loader), dice_scores

def test(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    dice_scores = []
    with torch.no_grad():
        with tqdm(loader, desc='Testing') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                dice_score = dice_coefficient(output, target)
                dice_scores.append(dice_score.item())
                pbar.set_postfix({'loss': loss.item(), 'dice': dice_score.item()})
    return total_loss / len(loader), dice_scores

def dice_coefficient(output, target):
    smooth = 1e-5
    output = (output > 0.5).float()
    intersection = (output * target).sum((1, 2, 3))
    union = output.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def save_losses(train_losses, val_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    df_train = pd.DataFrame({'Epoch': epochs, 'Loss': train_losses})
    df_val = pd.DataFrame({'Epoch': epochs, 'Loss': val_losses})
    df_train.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)
    df_val.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)

def save_dice_scores(dice_scores, save_path, filename):
    df = pd.DataFrame(dice_scores)
    df.to_excel(os.path.join(save_path, filename), index=False)

def plot_losses(train_losses, val_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))

def visualize_predictions(model, device, loader, save_path):
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(loader.dataset), 5, replace=False)
        fig, axs = plt.subplots(5, 3, figsize=(15, 20))
        for i, idx in enumerate(indices):
            data, target = loader.dataset[idx]
            original_idx = loader.dataset.indices[idx]
            data = data.unsqueeze(0).to(device)
            output = model(data)
            output = (output > 0.5).float()
            data = data.squeeze(0).cpu().numpy()
            target = target.cpu().numpy()
            output = output.squeeze(0).cpu().numpy()
            axs[i, 0].imshow(data[0], cmap='gray')
            axs[i, 0].set_title(loader.dataset.dataset.image_files[original_idx])
            axs[i, 1].imshow(target[0], cmap='gray')
            axs[i, 1].set_title('Ground Truth')
            axs[i, 2].imshow(output[0], cmap='gray')
            axs[i, 2].set_title('Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'predictions.png'))