# train.py
import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score  # approximate dice via 2*|A∩B|/(|A|+|B|)

def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc='Train', leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_epoch(model, loader, criterion, device, save_path, epoch):
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(tqdm(loader, desc='Validate', leave=False)):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()
            ds = dice_coeff(torch.sigmoid(preds), masks)
            dice_scores.append(ds.item())
    avg_loss = val_loss / len(loader)
    # Save dice scores per batch to Excel, one row per epoch:
    df = pd.DataFrame([dice_scores])
    fname = os.path.join(save_path, 'validation_dice_scores.xlsx')
    if epoch == 1:
        df.to_excel(fname, index=False, header=False)
    else:
        # append
        existing = pd.read_excel(fname, header=None)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_excel(fname, index=False, header=False)
    return avg_loss

def test_epoch(model, loader, device, save_path):
    model.eval()
    dice_scores = []
    img_names = []
    preds_all = []
    imgs_all = []
    masks_all = []
    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc='Test', leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            ds = dice_coeff(torch.sigmoid(logits), masks)
            dice_scores.append(ds.item())
            img_names.extend(names)
            imgs_all.append(imgs.cpu())
            masks_all.append(masks.cpu())
            preds_all.append(torch.sigmoid(logits).cpu())
    # save dice scores
    df = pd.DataFrame([dice_scores])
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False, header=False)
    return imgs_all, masks_all, preds_all, img_names

def plot_losses(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.title('Loss over epochs')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def visualize_predictions(imgs_all, masks_all, preds_all, img_names, save_path):
    import random
    idxs = random.sample(range(len(img_names)), k=5)
    fig, axes = plt.subplots(5, 3, figsize=(9,15))
    for row, i in enumerate(idxs):
        img = imgs_all[i][0].numpy()
        mask = masks_all[i][0].numpy()
        pred = (preds_all[i][0].numpy() > 0.5).astype(np.uint8)
        for col, data in enumerate([img, mask, pred]):
            ax = axes[row, col]
            ax.imshow(data, cmap='gray')
            if row == 0:
                ax.set_title(['Input Image','Ground Truth','Prediction'][col])
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(img_names[i], rotation=0, labelpad=50, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'))
    plt.close()
