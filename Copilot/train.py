import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

def dice_coefficient(pred, target):
    smooth = 1.
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, save_path):
    train_losses = []
    val_losses = []
    val_dice_scores = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
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
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                dice_scores.append(dice_coefficient(outputs, masks).item())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_dice_scores.append(dice_scores)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    pd.DataFrame({'Epoch': list(range(1, num_epochs+1)), 'Train Loss': train_losses}).to_excel(f"{save_path}/train_losses.xlsx", index=False)
    pd.DataFrame({'Epoch': list(range(1, num_epochs+1)), 'Val Loss': val_losses}).to_excel(f"{save_path}/val_losses.xlsx", index=False)
    pd.DataFrame(val_dice_scores).to_excel(f"{save_path}/validation_dice_scores.xlsx", index=False)

    torch.save(model.state_dict(), f"{save_path}/unet_model.pth")
    torch.save(model, f"{save_path}/unet_model_full.pth")

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

def test_model(model, test_loader, device, save_path):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores.append(dice_coefficient(outputs, masks).item())
    pd.DataFrame(dice_scores).to_excel(f"{save_path}/test_dice_scores.xlsx", index=False)

    # Visualize predictions
    import random
    random_samples = random.sample(list(test_loader), 5)
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    for i, (images, masks) in enumerate(random_samples):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        axes[i, 0].imshow(images[0].cpu().detach().numpy().squeeze(), cmap='gray')  # Select the first image in the batch
        axes[i, 0].set_title('Input Image')
        axes[i, 1].imshow(masks[0].cpu().detach().numpy().squeeze(), cmap='gray')  # Select the first mask in the batch
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(outputs[0].cpu().detach().numpy().squeeze(), cmap='gray')  # Select the first output in the batch
        axes[i, 2].set_title('Prediction')
    plt.savefig(f"{save_path}/predictions.png")