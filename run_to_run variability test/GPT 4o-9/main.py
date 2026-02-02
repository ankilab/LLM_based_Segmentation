# unet_segmentation/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchinfo import summary
from dataset import SegmentationDataset
from model import UNet
from train import *

if __name__ == "__main__":
    # Paths
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or use same_folder=True
    suffix = "_m"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-9"
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    lr = 1e-4
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, suffix=suffix, same_folder=False)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Train Loader: {len(train_loader)}, Val Loader: {len(val_loader)}, Test Loader: {len(test_loader)}")

    # Model
    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training Loop
    train_losses, val_losses, val_dices = [], [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, save_path, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time):.2f} seconds")

    # Save results
    save_losses(train_losses, "train_losses.xlsx", save_path)
    save_losses(val_losses, "val_losses.xlsx", save_path)
    save_dice_scores(val_dices, "validation_dice_scores.xlsx", save_path)
    plot_losses(train_losses, val_losses, save_path)
    torch.save(model, os.path.join(save_path, "unet_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_state_dict.pth"))

    # Testing
    test_dices = test(model, test_loader, device, save_path)
    save_dice_scores(test_dices, "test_dice_scores.xlsx", save_path)

    # Visualize Predictions
    visualize_predictions(model, test_loader, save_path, device)
