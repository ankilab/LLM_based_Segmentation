# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas as pd
import random

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def train_model(model, train_loader, val_loader, device, save_path, epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    val_dice_epochwise = []

    model.to(device)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] - Training")
        for images, masks, _ in train_pbar:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        dice_scores = []
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch}/{epochs}] - Validation")
        with torch.no_grad():
            for images, masks, _ in val_pbar:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()

                batch_dice = dice_score(preds, masks)
                dice_scores.append(batch_dice.cpu().numpy())
                val_pbar.set_postfix(loss=loss.item(), dice=np.mean(batch_dice.cpu().numpy()))

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_epochwise.append(np.array(dice_scores))

    # Save model
    torch.save(model, f"{save_path}/unet_model.pth")
    torch.save(model.state_dict(), f"{save_path}/unet_state_dict.pth")

    # Save Excel
    pd.DataFrame([train_losses], index=["loss"], columns=range(1, epochs + 1)).to_excel(f"{save_path}/train_losses.xlsx")
    pd.DataFrame([val_losses], index=["loss"], columns=range(1, epochs + 1)).to_excel(f"{save_path}/val_losses.xlsx")
    dice_array = np.array([np.mean(batch, axis=0) for batch in val_dice_epochwise])
    pd.DataFrame(dice_array).to_excel(f"{save_path}/validation_dice_scores.xlsx", index=False)

    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end) / 1000
    print(f"\nTotal Training Time: {total_time:.2f} seconds")

    return train_losses, val_losses

def test_model(model, test_loader, device, save_path):
    model.eval()
    all_dice = []
    sample_visuals = []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for images, masks, filenames in test_pbar:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice = dice_score(preds, masks)
            all_dice.append(dice.cpu().numpy())

            for i in range(len(images)):
                sample_visuals.append((images[i].cpu(), masks[i].cpu(), preds[i].cpu(), filenames[i]))

    # Save Dice Scores
    pd.DataFrame(all_dice).to_excel(f"{save_path}/test_dice_scores.xlsx", index=False)

    # Plot Predictions
    indices = random.sample(range(len(sample_visuals)), 5)
    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    for i, idx in enumerate(indices):
        img, mask, pred, fname = sample_visuals[idx]
        axs[i, 0].imshow(img.squeeze(), cmap='gray')
        axs[i, 0].set_title(f"{fname}")
        axs[i, 1].imshow(mask.squeeze(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(pred.squeeze() > 0.5, cmap='gray')
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}/test_predictions.png")

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(f"{save_path}/loss_plot.png")
