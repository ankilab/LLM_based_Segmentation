import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    train_losses = []
    val_losses = []
    val_dice_scores = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                dice_score = dice_coefficient(outputs, masks)
                dice_scores.append(dice_score)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_dice_scores.append(dice_scores)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    save_losses(train_losses, val_losses, save_path)
    save_dice_scores(val_dice_scores, save_path, "validation_dice_scores.xlsx")
    plot_losses(train_losses, val_losses, save_path)
    torch.save(model.state_dict(), f"{save_path}/unet_model.pth")
    torch.save(model, f"{save_path}/unet_model_full.pth")


def dice_coefficient(pred, target):
    smooth = 1.0
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def save_losses(train_losses, val_losses, save_path):
    train_df = pd.DataFrame([train_losses], columns=[f"Epoch {i + 1}" for i in range(len(train_losses))])
    val_df = pd.DataFrame([val_losses], columns=[f"Epoch {i + 1}" for i in range(len(val_losses))])
    train_df.to_excel(f"{save_path}/train_losses.xlsx", index=False)
    val_df.to_excel(f"{save_path}/val_losses.xlsx", index=False)


def save_dice_scores(dice_scores, save_path, filename):
    dice_df = pd.DataFrame(dice_scores)
    dice_df.to_excel(f"{save_path}/{filename}", index=False)


# def plot_losses(train_losses, val_losses, save_path):
#     plt.figure()
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
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

def test(model, test_loader, device, save_path):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing') as pbar:
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice = dice_coefficient(outputs, masks)
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