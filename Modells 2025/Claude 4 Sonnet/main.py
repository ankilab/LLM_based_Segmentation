import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet, CombinedLoss
from train import train_model, test_model, plot_losses, visualize_predictions


def main():
    # Configuration
    DATA_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"  # Change this to your dataset path
    SAVE_PATH = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Claude 4 Sonnet\\out of the box\\BAGLS output"  # Change this to your desired save path
    MASK_SUFFIX = "_seg"  # Change this if your mask suffix is different

    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = (256, 256)

    # Create save directory
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    print("Loading dataset...")
    dataset = SegmentationDataset(
        image_dir=DATA_DIR,
        mask_suffix=MASK_SUFFIX,
        image_size=IMAGE_SIZE
    )

    # Split dataset
    total_size = len(dataset)
    indices = list(range(total_size))

    # First split: 80% train, 20% temp (which will be split into 10% val, 10% test)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )

    # Second split: 10% val, 10% test from the 20% temp
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, shuffle=True
    )

    print(f"Dataset split:")
    print(f"  Training samples: {len(train_indices)} ({len(train_indices) / total_size * 100:.1f}%)")
    print(f"  Validation samples: {len(val_indices)} ({len(val_indices) / total_size * 100:.1f}%)")
    print(f"  Test samples: {len(test_indices)} ({len(test_indices) / total_size * 100:.1f}%)")

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Data loader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Initialize model
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.5)  # Balanced BCE and Dice loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=SAVE_PATH
    )

    # Plot losses
    plot_losses(train_losses, val_losses, SAVE_PATH)
    print(f"Loss plot saved to {os.path.join(SAVE_PATH, 'loss_plot.png')}")

    # Testing
    print("\n" + "=" * 50)
    print("STARTING TESTING")
    print("=" * 50)

    test_dice_scores, test_results = test_model(
        model=model,
        dataloader=test_loader,
        device=device,
        save_path=SAVE_PATH
    )

    print(f"Test Results:")
    print(f"  Mean Dice Score: {np.mean(test_dice_scores):.4f}")
    print(f"  Std Dice Score: {np.std(test_dice_scores):.4f}")
    print(f"  Min Dice Score: {np.min(test_dice_scores):.4f}")
    print(f"  Max Dice Score: {np.max(test_dice_scores):.4f}")

    # Visualize predictions
    visualize_predictions(test_results, SAVE_PATH, num_samples=5)
    print(f"Test predictions visualization saved to {os.path.join(SAVE_PATH, 'test_predictions.png')}")

    print("\n" + "=" * 50)
    print("TRAINING AND TESTING COMPLETED")
    print("=" * 50)
    print(f"All results saved to: {SAVE_PATH}")
    print("Files created:")
    print("  - complete_model.pth (full model)")
    print("  - model_state_dict.pth (model weights)")
    print("  - train_losses.xlsx (training losses)")
    print("  - val_losses.xlsx (validation losses)")
    print("  - validation_dice_scores.xlsx (validation dice scores)")
    print("  - test_dice_scores.xlsx (test dice scores)")
    print("  - loss_plot.png (loss visualization)")
    print("  - test_predictions.png (prediction visualization)")


if __name__ == "__main__":
    main()