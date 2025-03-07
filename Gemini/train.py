import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from dataset import SegmentationDataset
from model import Unet
import torch.nn.functional as F

def train(model, train_loader, val_loader, num_epochs, learning_rate, save_path):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    val_dice_scores = []
    test_dice_scores = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", leave=False):
            optimizer.zero_grad()
            outputs = model(images)
            # Resize the target masks to match the output size
            # masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_dice = 0
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)", leave=False):
                outputs = model(images)

                # Resize the target masks for validation as well
                # masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)

                val_loss = criterion(outputs, masks)
                epoch_val_loss += val_loss.item()
                epoch_val_dice += dice_coeff(outputs, masks)
            epoch_val_loss /= len(val_loader)
            epoch_val_dice /= len(val_loader)
            val_losses.append(epoch_val_loss)
            val_dice_scores.append(epoch_val_dice)

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}")

    # Save losses and dice scores
    pd.DataFrame(train_losses).to_excel(os.path.join(save_path, "train_losses.xlsx"))
    pd.DataFrame(val_losses).to_excel(os.path.join(save_path, "val_losses.xlsx"))
    pd.DataFrame(val_dice_scores).to_excel(os.path.join(save_path, "validation_dice_scores.xlsx"))

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model.pth"))

    # Calculate and print training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Visualize losses
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))


def dice_coeff(pred, target):
    smooth = 1.0

    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Convert probabilities to binary using thresholding (e.g., 0.5)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()
