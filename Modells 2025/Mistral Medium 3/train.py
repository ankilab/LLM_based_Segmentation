import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from torchinfo import summary

def dice_coeff(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_losses = []
    val_losses = []
    val_dice_scores = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop with tqdm
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', leave=True)
        for images, masks in train_loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        epoch_dice_scores = []

        val_loop = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', leave=True)
        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() * images.size(0)

                # Calculate Dice score
                dice = dice_coeff(outputs, masks)
                epoch_dice_scores.append(dice.item())

                val_loop.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_dice_scores.append(epoch_dice_scores)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

    # Save losses to Excel
    train_loss_df = pd.DataFrame([range(1, num_epochs + 1), train_losses]).T
    train_loss_df.columns = ['Epoch', 'Train Loss']
    train_loss_df.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)

    val_loss_df = pd.DataFrame([range(1, num_epochs + 1), val_losses]).T
    val_loss_df.columns = ['Epoch', 'Val Loss']
    val_loss_df.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)

    # Save validation Dice scores
    dice_df = pd.DataFrame(val_dice_scores)
    dice_df.to_excel(os.path.join(save_path, 'validation_dice_scores.xlsx'), index=False)

    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model.pth'))
    torch.save(model, os.path.join(save_path, 'unet_model_full.pth'))

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    # Calculate total training time
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    return model, train_losses, val_losses, val_dice_scores

def test_model(model, test_loader, criterion, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_dice_scores = []

    test_loop = tqdm(test_loader, desc='Testing', leave=True)
    with torch.no_grad():
        for images, masks in test_loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            # Calculate Dice score
            dice = dice_coeff(outputs, masks)
            test_dice_scores.append(dice.item())

            test_loop.set_postfix(loss=loss.item())

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # Save test Dice scores
    dice_df = pd.DataFrame(test_dice_scores).T
    dice_df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)

    return test_loss, test_dice_scores

def visualize_predictions(model, test_loader, save_path, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Get random samples
    indices = torch.randperm(len(test_loader.dataset))[:num_samples]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    sample_loader = DataLoader(test_loader.dataset, batch_size=1, sampler=sampler)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    fig.suptitle('Input Image | Ground Truth | Prediction', fontsize=16)

    with torch.no_grad():
        for i, (images, masks) in enumerate(sample_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_mask = (outputs > 0.5).float().squeeze().cpu().numpy()
            input_img = images.squeeze().cpu().numpy()
            gt_mask = masks.squeeze().cpu().numpy()

            # Plot input image
            axes[i, 0].imshow(input_img, cmap='gray')
            axes[i, 0].set_title(f"Input: {test_loader.dataset.image_files[indices[i]]}")
            axes[i, 0].axis('off')

            # Plot ground truth mask
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            # Plot predicted mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'))
    plt.close()