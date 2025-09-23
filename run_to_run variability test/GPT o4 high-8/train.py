# train.py

import os
import time
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score  # approximate dice via IoU

def dice_coeff(pred, target, eps=1e-6):
    pred_flat = pred.view(-1).cpu().numpy()
    targ_flat = target.view(-1).cpu().numpy()
    # dice = 2 * intersection / (sum)
    return (2. * (pred_flat * targ_flat).sum() + eps) / (pred_flat.sum() + targ_flat.sum() + eps)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_dices = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Validate", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
            # per-batch dice
            preds = (outputs > 0.5).float()
            dice = dice_coeff(preds, masks)
            all_dices.append(dice)
    return running_loss / len(loader.dataset), all_dices

def test_model(model, loader, device):
    model.eval()
    all_dices = []
    samples = []
    with torch.no_grad():
        for imgs, masks, fnames in tqdm(loader, desc="Test", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            dice = dice_coeff(preds, masks)
            all_dices.append(dice)
            samples.extend(zip(imgs.cpu(), masks.cpu(), preds.cpu(), fnames))
    return all_dices, samples

def save_losses(losses, path, filename, header):
    df = pd.DataFrame([losses], index=[0])
    df.columns = [f"Epoch_{i+1}" for i in range(len(losses))]
    df.to_excel(os.path.join(path, filename), index=False)

def save_batch_dices(all_epoch_batches, path, filename):
    # rows = epochs, cols = batch_i
    df = pd.DataFrame(all_epoch_batches)
    df.index = [f"Epoch_{i+1}" for i in range(df.shape[0])]
    df.columns = [f"Batch_{j+1}" for j in range(df.shape[1])]
    df.to_excel(os.path.join(path, filename))

def plot_losses(train_losses, val_losses, path):
    plt.figure()
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses, label='Val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(path, 'loss_curve.png'))
    plt.close()
