import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import os


def dice_score(preds, targets, threshold=0.5):
    smooth = 1e-6
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def save_losses_to_excel(path, losses, file_name):
    df = pd.DataFrame(losses, index=["Epoch " + str(i + 1) for i in range(len(losses))])
    df.to_excel(os.path.join(path, file_name), index=True)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_path, device="cuda"):
    train_losses, val_losses = [], []
    dice_scores = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_dice_scores = []
        epoch_start_time = time.time()

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                running_val_loss += val_loss.item()

                batch_dice = dice_score(outputs, masks)
                epoch_dice_scores.append(batch_dice.item())

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        dice_scores.append(epoch_dice_scores)

        save_losses_to_excel(save_path, dice_scores, "validation_dice_scores.xlsx")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} completed in {(epoch_end_time - epoch_start_time) / 60:.2f} minutes")

    # Save the losses and the model
    save_losses_to_excel(save_path, train_losses, "train_losses.xlsx")
    save_losses_to_excel(save_path, val_losses, "val_losses.xlsx")

    torch.save(model, os.path.join(save_path, "unet_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_state.pth"))

    plot_losses(train_losses, val_losses, save_path)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "losses_plot.png"))
    plt.show()


def test_model(model, test_loader, save_path, device="cuda"):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_dice = dice_score(outputs, masks)
            dice_scores.append(batch_dice.item())

    save_losses_to_excel(save_path, dice_scores, "test_dice_scores.xlsx")


def visualize_predictions(model, test_loader, save_path, device="cuda"):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i == 5:  # Show 5 samples
                break
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            plt.figure(figsize=(12, 15))
            for idx in range(5):
                plt.subplot(5, 3, idx * 3 + 1)
                plt.imshow(images[idx].cpu().squeeze(), cmap="gray")
                plt.title("Input Image")
                plt.subplot(5, 3, idx * 3 + 2)
                plt.imshow(masks[idx].cpu().squeeze(), cmap="gray")
                plt.title("Ground Truth")
                plt.subplot(5, 3, idx * 3 + 3)
                plt.imshow(outputs[idx].cpu().squeeze(), cmap="gray")
                plt.title("Prediction")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"predictions_{i + 1}.png"))
            plt.show()
