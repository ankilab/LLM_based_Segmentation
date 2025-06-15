import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchinfo import summary
from tqdm import tqdm
import time
from dataset import SegmentationDataset
from model import UNet
from train import *

# Configuration
DATA_PATH = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
MASK_PATH = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'
SAVE_PATH = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek R1\\out of the box\\Brain output'
os.makedirs(SAVE_PATH, exist_ok=True)

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)


def main():
    # Create dataset
    dataset = SegmentationDataset(DATA_PATH, mask_dir= MASK_PATH, img_size=IMG_SIZE)

    # Split indices
    idx = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=SEED)

    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create subsets and dataloaders
    train_set = data.Subset(dataset, train_idx)
    val_set = data.Subset(dataset, val_idx)
    test_set = data.Subset(dataset, test_idx)

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE)

    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Print model summary
    summary(model, input_size=(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Loss and optimizer
    criterion = DiceBCELoss(weight=0.7)  # 70% BCE, 30% Dice
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    #optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    train_losses = []
    val_losses = []
    val_dice_scores = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        train_loss = train_epoch(model, train_loader, DEVICE, optimizer, criterion)
        train_losses.append(train_loss)

        # Validation
        val_loss, dice_scores = validate_epoch(model, val_loader, DEVICE, criterion)
        val_losses.append(val_loss)
        val_dice_scores.append(dice_scores)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # Save losses
    pd.DataFrame(train_losses).to_excel(os.path.join(SAVE_PATH, 'train_losses.xlsx'), header=False)
    pd.DataFrame(val_losses).to_excel(os.path.join(SAVE_PATH, 'val_losses.xlsx'), header=False)

    # Save dice scores (epochs x batches)
    # dice_df = pd.DataFrame(val_dice_scores).T
    # dice_df.columns = [f'Epoch {i + 1}' for i in range(EPOCHS)]
    # dice_df.to_excel(os.path.join(SAVE_PATH, 'validation_dice_scores.xlsx'))

    # manual edit: Save validation dice scores (rows=epochs, cols=batches)
    dice_df = pd.DataFrame(
        val_dice_scores,
        index=[f'Epoch {i + 1}' for i in range(EPOCHS)]
    )
    dice_df.columns = [f'Batch {j + 1}' for j in range(dice_df.shape[1])]
    dice_df.index.name = 'Epoch'
    dice_df.to_excel(
        os.path.join(SAVE_PATH, 'validation_dice_scores.xlsx'),
        index = True
    )


    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'unet_model.pth'))
    torch.save(model, os.path.join(SAVE_PATH, 'unet_full.pth'))

    # Loss plot
    save_loss_plot(train_losses, val_losses, SAVE_PATH)

    # Testing
    test_dice, test_samples = test_model(model, test_loader, DEVICE)
    print(f"Test Dice: {np.mean(test_dice):.4f} ± {np.std(test_dice):.4f}")

    # Save test dice scores
    pd.DataFrame(test_dice).to_excel(os.path.join(SAVE_PATH, 'test_dice_scores.xlsx'), header=False)
    # Save test dice scores as one “Test” row, batches as columns
    # manual edit:
    test_df = pd.DataFrame([test_dice], index=['Test'])
    test_df.columns = [f'Batch {j + 1}' for j in range(test_df.shape[1])]
    test_df.index.name = 'Split'
    test_df.to_excel(
        os.path.join(SAVE_PATH, 'test_dice_scores.xlsx'),
        index=True
    )

    # Visualize samples
    save_sample_visualization(test_samples, SAVE_PATH)


if __name__ == "__main__":
    main()