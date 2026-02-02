import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random
from tqdm import tqdm
from model import dice_coefficient


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return running_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model for one epoch and return loss and dice scores per batch"""
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Calculate dice score for this batch
            dice_score = dice_coefficient(outputs, masks)
            dice_scores.append(dice_score.item())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice_score.item():.4f}'})

    avg_loss = running_loss / len(dataloader)
    return avg_loss, dice_scores


def test_model(model, dataloader, device, save_path):
    """Test the model and return dice scores per batch"""
    model.eval()
    dice_scores = []
    test_results = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for batch_idx, (images, masks, filenames) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # Calculate dice score for this batch
            dice_score = dice_coefficient(outputs, masks)
            dice_scores.append(dice_score.item())

            # Store results for visualization
            for i in range(images.size(0)):
                test_results.append({
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'prediction': outputs[i].cpu(),
                    'filename': filenames[i]
                })

            pbar.set_postfix({'Dice': f'{dice_score.item():.4f}'})

    # Save dice scores to Excel
    dice_df = pd.DataFrame([dice_scores], columns=[f'Batch_{i + 1}' for i in range(len(dice_scores))])
    dice_df.to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)

    return dice_scores, test_results


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_path):
    """Complete training procedure"""

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    train_losses = []
    val_losses = []
    val_dice_scores_all = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation
        val_loss, val_dice_scores = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dice_scores_all.append(val_dice_scores)

        # Step scheduler
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Mean Val Dice: {np.mean(val_dice_scores):.4f}")

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")

    # Save losses to Excel files
    train_df = pd.DataFrame([list(range(1, num_epochs + 1)), train_losses],
                            index=['Epoch', 'Training_Loss']).T
    train_df.to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False)

    val_df = pd.DataFrame([list(range(1, num_epochs + 1)), val_losses],
                          index=['Epoch', 'Validation_Loss']).T
    val_df.to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False)

    # Save validation dice scores
    val_dice_df = pd.DataFrame(val_dice_scores_all,
                               columns=[f'Batch_{i + 1}' for i in range(len(val_dice_scores_all[0]))])
    val_dice_df.insert(0, 'Epoch', list(range(1, num_epochs + 1)))
    val_dice_df.to_excel(os.path.join(save_path, 'validation_dice_scores.xlsx'), index=False)

    # Save model
    torch.save(model, os.path.join(save_path, 'complete_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict.pth'))

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, save_path):
    """Plot and save training and validation losses"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.title('Training and Validation Losses Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_predictions(test_results, save_path, num_samples=5):
    """Visualize predictions for random test samples"""

    # Select random samples
    random_indices = random.sample(range(len(test_results)), min(num_samples, len(test_results)))
    selected_results = [test_results[i] for i in random_indices]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle('Test Results: Input Image, Ground Truth, Prediction', fontsize=16)

    for i, result in enumerate(selected_results):
        image = result['image'].squeeze().numpy()
        mask = result['mask'].squeeze().numpy()
        prediction = result['prediction'].squeeze().numpy()
        filename = result['filename']

        # Input Image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Input Image\n{filename}')
        axes[i, 0].axis('off')

        # Ground Truth
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth\n{filename}')
        axes[i, 1].axis('off')

        # Prediction
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title(f'Prediction\n{filename}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()