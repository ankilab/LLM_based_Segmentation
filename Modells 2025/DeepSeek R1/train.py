import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# Loss function with background weighting
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.7):
        super(DiceBCELoss, self).__init__()
        self.weight = weight  # Weight for BCE (Dice weight = 1 - weight)

    def forward(self, inputs, targets):
        # Binary Cross Entropy
        bce = nn.functional.binary_cross_entropy(inputs, targets)

        # Dice Coefficient
        smooth = 1.0
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return self.weight * bce + (1 - self.weight) * (1 - dice)


def dice_score(preds, targets):
    """
    Compute per-sample Dice, with smoothing to avoid 0/0=NaN when
    both pred and target are all zeros.
    """
    preds_bin = (preds > 0.5).float()
    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    smooth = 1e-6
    return (2. * intersection + smooth) / (union + smooth)


def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, device, criterion, save_path=None, epoch_idx=None):
    model.eval()
    running_loss = 0.0
    batch_means  = []

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            bd = dice_score(outputs, masks).cpu().numpy()
            batch_means.append(bd.mean())

    epoch_loss = running_loss / len(loader.dataset)
    batch_means = np.array(batch_means)

    if save_path is not None and epoch_idx is not None:
        df = pd.DataFrame(
            [batch_means],
            columns=[f'Batch {i+1}' for i in range(len(batch_means))]
        )
        df.index.name = 'Epoch'
        df.to_excel(
            os.path.join(save_path, f'val_dice_epoch_{epoch_idx+1}.xlsx'),
            index=True
        )

    return epoch_loss, batch_means


def test_model(model, loader, device, save_path=None):
    model.eval()
    batch_means = []
    all_images, all_masks, all_preds, all_names = [], [], [], []

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Testing"):
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            bd = dice_score(outputs, masks).cpu().numpy()
            batch_means.append(bd.mean())

            if len(all_images) < 5:
                all_images.append(images.cpu())
                all_masks .append(masks.cpu())
                all_preds .append(outputs.cpu())
                all_names .extend(names)

    vis_images = torch.cat(all_images, dim=0)[:5]
    vis_masks  = torch.cat(all_masks,  dim=0)[:5]
    vis_preds  = torch.cat(all_preds,  dim=0)[:5]

    batch_means = np.array(batch_means)

    if save_path is not None:
        df = pd.DataFrame(
            [batch_means],
            columns=[f'Batch {i+1}' for i in range(len(batch_means))]
        )
        df.index.name = 'Split'
        df.index = ['Test']
        df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=True)

    return batch_means, (vis_images, vis_masks, vis_preds, all_names[:5])


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses,   label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()


def save_sample_visualization(samples, save_path):
    images, masks, preds, names = samples
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    fig.suptitle('Segmentation Results', fontsize=16)

    for i in range(5):
        img  = images[i].squeeze().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Input: {names[i]}")

        mask = masks[i].squeeze().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")

        pred = preds[i].squeeze().numpy()
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sample_predictions.png'))
    plt.close()
