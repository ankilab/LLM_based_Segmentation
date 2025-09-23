import os
import time
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score


def dice_coeff(pred, target, eps=1e-6):
    """Compute Dice coefficient per batch"""
    pred = (pred > 0.5).float().contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2. * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    loop = tqdm(dataloader, desc='Train', leave=False)
    for imgs, masks, _ in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loop.set_postfix(loss=float(loss.item()))
    return sum(losses) / len(losses)


def validate_epoch(model, dataloader, criterion, device, save_path, epoch):
    model.eval()
    losses, all_dices = [], []
    batch_dices = []
    loop = tqdm(dataloader, desc='Validate', leave=False)
    for i, (imgs, masks, _) in enumerate(loop):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(imgs))
            loss = criterion(preds, masks)
        losses.append(loss.item())
        dices = dice_coeff(preds, masks).cpu().numpy()
        batch_dices.append(dices)
        all_dices.extend(dices)
        loop.set_postfix(loss=float(loss.item()), dice=float(dices.mean()))
    # save dice per batch to Excel
    df = pd.DataFrame(batch_dices)
    df.index += 1
    df.to_excel(os.path.join(save_path, 'validation_dice_scores.xlsx'), index_label='Epoch')
    return sum(losses) / len(losses), sum(all_dices) / len(all_dices)


def test_model(model, dataloader, criterion, device, save_path):
    model.eval()
    all_dices = []
    batch_dices = []
    loop = tqdm(dataloader, desc='Test', leave=False)
    results = []
    for i, (imgs, masks, fnames) in enumerate(loop):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(imgs))
            loss = criterion(preds, masks)
        dices = dice_coeff(preds, masks).cpu().numpy()
        batch_dices.append(dices)
        all_dices.extend(dices)
        results.append((imgs.cpu(), masks.cpu(), preds.cpu(), fnames))
        loop.set_postfix(loss=float(loss.item()), dice=float(dices.mean()))
    # save dice
    df = pd.DataFrame(batch_dices)
    df.index += 1
    df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index_label='Batch')
    return results


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()


def visualize_predictions(results, save_path, num_samples=5):
    import matplotlib.pyplot as plt
    import numpy as np

    samples = random.sample(results, k=min(num_samples, len(results)))
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3*len(samples)))
    for i, (imgs, masks, preds, fnames) in enumerate(samples):
        img = imgs[0,0].numpy()
        gt  = masks[0,0].numpy()
        pr  = (preds[0,0].numpy() > 0.5).astype(float)
        for j, arr, title in zip([0,1,2],[img,gt,pr], ['Input','Ground Truth','Prediction']):
            ax = axes[i,j]
            ax.imshow(arr, cmap='gray')
            if i == 0:
                ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Sample: {fnames[0]}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'))
    plt.close()


def save_epoch_losses(epoch_losses, filename):
    """
    epoch_losses: list of floats
    saves two-row Excel: first row epochs, second row losses
    """
    df = pd.DataFrame([list(range(1, len(epoch_losses)+1)), epoch_losses])
    df.to_excel(filename, header=False, index=False)
