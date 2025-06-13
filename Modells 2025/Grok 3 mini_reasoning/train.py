import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score  # For Dice score (F1 for binary)


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()  # Threshold predictions
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device='cuda',
                save_path='results/'):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary segmentation loss
    train_losses = []
    val_losses = []
    val_dice_scores = []  # List of lists: [epoch1: [batch1_dice, batch2_dice, ...], ...]

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for batch in train_progress:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            #loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dim for masks
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_progress.set_postfix({'Loss': loss.item()})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = []  # Per batch Dice scores
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]')

        with torch.no_grad():
            for batch in val_progress:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                epoch_val_loss += loss.item()

                # Calculate Dice per batch
                batch_dice = dice_score(outputs.cpu(), masks.unsqueeze(1).cpu())
                epoch_val_dice.append(batch_dice.item())

                val_progress.set_postfix({'Loss': loss.item(), 'Dice': batch_dice.item()})

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(epoch_val_dice)  # Store per epoch

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    total_time = time.time() - start_time
    print(f'Total Training Time: {total_time:.2f} seconds')

    # Save losses to Excel
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Average Train Loss': train_losses}).to_excel(
        os.path.join(save_path, 'train_losses.xlsx'), index=False)
    pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Average Val Loss': val_losses}).to_excel(
        os.path.join(save_path, 'val_losses.xlsx'), index=False)

    # Save val Dice scores: epochs as rows, batches as columns
    val_dice_df = pd.DataFrame(val_dice_scores)  # Each row is an epoch, columns are batches
    val_dice_df.to_excel(os.path.join(save_path, 'validation_dice_scores.xlsx'), index=False)

    # Save model
    torch.save(model, os.path.join(save_path, 'model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict.pth'))

    # Visualize losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(save_path, 'losses_plot.png'))  # Save as PNG
    plt.close()


def test_model(model, test_loader, device='cuda', save_path='results/'):
    model.eval()
    test_dice_scores = []  # Per batch Dice scores
    all_images = []  # For visualization
    all_masks = []  # For visualization
    all_preds = []  # For visualization
    filenames = []  # For visualization

    test_progress = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for batch in test_progress:
            images, masks = batch
            filenames.extend(
                [f for f in test_loader.dataset.image_files if f in test_loader.dataset.samples])  # Mock for filenames
            images = images.to(device)
            outputs = model(images)
            batch_dice = dice_score(outputs.cpu(), masks.unsqueeze(1).cpu())
            test_dice_scores.append(batch_dice.item())

            # Store for visualization
            all_images.append(images.cpu())
            all_masks.append(masks.unsqueeze(1).cpu())  # Add channel dim
            all_preds.append((outputs.cpu() > 0.5).float())

    # Save test Dice scores
    pd.DataFrame([test_dice_scores]).to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'),
                                              index=False)  # One row for test

    # Visualize 5 random samples
    indices = np.random.choice(len(all_images[0]), 5, replace=False)  # 5 random from first batch
    plt.figure(figsize=(15, 25))  # 5 rows, 3 columns
    for i, idx in enumerate(indices):
        plt.subplot(5, 3, i * 3 + 1)
        plt.imshow(all_images[0][idx].squeeze(), cmap='gray')
        plt.title(f'Input Image\n{filenames[idx]}')
        plt.axis('off')

        plt.subplot(5, 3, i * 3 + 2)
        plt.imshow(all_masks[0][idx].squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(5, 3, i * 3 + 3)
        plt.imshow(all_preds[0][idx].squeeze(), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

    plt.suptitle('Test Predictions for 5 Random Samples')
    plt.savefig(os.path.join(save_path, 'test_visualization.png'))  # Save as PNG
    plt.close()