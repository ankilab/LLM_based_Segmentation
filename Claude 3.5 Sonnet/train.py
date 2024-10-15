import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from model import UNet
from dataset import SegmentationDataset, get_transform


def dice_coeff(pred, target):
    smooth = 1e-5
    num = pred.size(0)
    pred = pred.view(num, -1)
    target = target.view(num, -1)
    intersection = (pred * target).sum(1)
    return (2. * intersection + smooth) / (pred.sum(1) + target.sum(1) + smooth)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    train_losses = []
    val_losses = []
    val_dice_scores = []

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs} (Train)') as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        dice_scores = []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)') as pbar:
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    dice = dice_coeff(outputs, masks)
                    dice_scores.extend(dice.tolist())
                    pbar.update(1)
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.mean().item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(dice_scores)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
            torch.save(model, f'{save_path}/best_model_full.pth')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    # Save losses
    pd.DataFrame([range(1, num_epochs + 1), train_losses]).T.to_excel(f'{save_path}/train_losses.xlsx', index=False,
                                                                      header=['Epoch', 'Loss'])
    pd.DataFrame([range(1, num_epochs + 1), val_losses]).T.to_excel(f'{save_path}/val_losses.xlsx', index=False,
                                                                    header=['Epoch', 'Loss'])

    # Save validation dice scores
    pd.DataFrame(val_dice_scores).to_excel(f'{save_path}/validation_dice_scores.xlsx', index=False)

    # Plot losses
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

def test(model, test_loader, device, save_path):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing') as pbar:
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice = dice_coeff(outputs, masks)
                dice_scores.extend(dice.tolist())
                pbar.update(1)
                pbar.set_postfix({'Dice': f'{dice.mean().item():.4f}'})

    pd.DataFrame(dice_scores).to_excel(f'{save_path}/test_dice_scores.xlsx', index=False)

    return dice_scores


def visualize_predictions(model, dataloader, device, save_path):
    model.eval()
    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    axs = axs.ravel()

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i == 5:
                break
            # Ensure images are 4D
            images = images.to(device)
            masks = masks.to(device)

            # Ensure images have the batch dimension (if not already)
            if images.dim() == 3:
                images = images.unsqueeze(0)

            outputs = model(images)
            outputs = (outputs > 0.5).float()

            # Plot input image, ground truth, and prediction
            axs[i * 3].imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 1].imshow(masks[0].cpu().numpy().squeeze(), cmap='gray')
            axs[i * 3 + 2].imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')

    plt.savefig(f'{save_path}/predictions.png')
    plt.close()
