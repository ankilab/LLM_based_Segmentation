import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchinfo import summary
import time
import numpy as np
from dataset import SegmentationDataset, get_transforms
from model import UNet
from train import (
    BCEDiceLoss, train_epoch, validate_epoch,
    save_losses, save_dice_scores, plot_losses,
    save_model, visualize_results
)


def main():
    # Configuration
    data_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'  # Update this path
    #image_dir = os.path.join(data_dir, 'images')
    #mask_dir = os.path.join(data_dir, 'masks')
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    mask_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek V3\\out of the box\\BAGLS output'  # Directory to save results
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_epochs = 25
    learning_rate = 0.001
    image_size = (256, 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and Transforms
    transform = get_transforms(image_size)
    dataset = SegmentationDataset(image_dir, mask_dir, transform)

    # Split dataset
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(
        test_idx, test_size=0.5, random_state=42)

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Print model summary
    summary(model, input_size=(batch_size, 1, *image_size))

    # Loss and optimizer
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    val_dice_scores = []

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        train_losses.append(train_loss)

        # Validate
        val_loss, dice_scores = validate_epoch(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_dice_scores.append(dice_scores)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")

    # Save results
    save_losses(train_losses, val_losses, save_path)
    save_dice_scores(val_dice_scores, 'validation_dice_scores.xlsx', save_path)
    plot_losses(train_losses, val_losses, save_path)
    save_model(model, save_path)

    # Test
    test_loss, test_dice_scores = validate_epoch(model, test_loader, device, criterion)
    save_dice_scores([test_dice_scores], 'test_dice_scores.xlsx', save_path)
    print(f"\nTest Loss: {test_loss:.4f}, Avg Dice: {np.mean(test_dice_scores):.4f}")

    # Visualize results
    visualize_results(model, test_loader, device, save_path)


if __name__ == "__main__":
    main()