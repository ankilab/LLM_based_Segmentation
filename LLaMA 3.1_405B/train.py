import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    with tqdm(loader, desc=f'Epoch {epoch+1}') as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
    with torch.no_grad():
        with tqdm(loader, desc=f'Validation Epoch {epoch+1}') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                dice_score = calculate_dice_score(outputs, masks)
                dice_scores.append(dice_score)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice_score': f'{dice_score:.4f}'})
    return total_loss / len(loader), dice_scores

def test(model, device, loader):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        with tqdm(loader, desc='Testing') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice_score = calculate_dice_score(outputs, masks)
                dice_scores.append(dice_score)
                pbar.set_postfix({'dice_score': f'{dice_score:.4f}'})
    return dice_scores

def calculate_dice_score(outputs, masks):
    outputs = (outputs > 0.5).float()
    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum()
    return 2 * intersection / union

def save_losses(train_losses, val_losses, save_path):
    train_df = pd.DataFrame(train_losses)
    val_df = pd.DataFrame(val_losses)
    train_df.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)
    val_df.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)

def visualize_losses(train_losses, val_losses, save_path):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))

def visualize_predictions(model, device, loader, save_path):
    model.eval()
    with torch.no_grad():
        images, masks, outputs = next(iter(loader))
        images, masks, outputs = images.to(device), masks.to(device), model(images)
        fig, ax = plt.subplots(5, 3, figsize=(15, 15))
        for i in range(5):
            ax[i, 0].imshow(images[i].cpu().numpy(), cmap='gray')
            ax[i, 0].set_title('Input Image')
            ax[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray')
            ax[i, 1].set_title('Ground Truth')
            ax[i, 2].imshow(outputs[i].cpu().numpy(), cmap='gray')
            ax[i, 2].set_title('Prediction')
        plt.savefig(os.path.join(save_path, 'predictions.png'))