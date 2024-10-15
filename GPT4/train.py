# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import numpy as np


def dice_score(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, dice_scores


def test(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    return np.mean(dice_scores)


def save_losses(train_losses, val_losses, save_path):
    pd.DataFrame(train_losses).to_excel(f"{save_path}/train_losses.xlsx", index=False)
    pd.DataFrame(val_losses).to_excel(f"{save_path}/val_losses.xlsx", index=False)

#
# def plot_losses(train_losses, val_losses, save_path):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.title('Training and Validation Losses Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f"{save_path}/loss_plot.png")
#     plt.close()

def plot_losses(train_losses, val_losses, save_path):
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

def save_dice_scores(dice_scores, save_path, file_name):
    df_dice = pd.DataFrame(dice_scores)
    df_dice.to_excel(f'{save_path}/{file_name}.xlsx', index=False)


def save_model(model, save_path):
    torch.save(model.state_dict(), f"{save_path}/model_state.pth")
    torch.save(model, f"{save_path}/model.pth")


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
            outputs = (outputs > 0.5).float()
            axs[i * 3].imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 1].imshow(masks[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 2].imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')

    plt.savefig(f'{save_path}/predictions.png')
    plt.close()