import os
import torch
from torch.utils.data import DataLoader
from dataset import GrayscaleBinaryMaskDataset
from model import UNet
from train import train_one_epoch, validate, test, plot_losses
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import time

if __name__ == '__main__':
    # Hyperparameters
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Skin cancer\subset\images'
    mask_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Skin cancer\subset\masks'
    mask_suffix = ''
    batch_size = 16
    epochs = 20
    lr = 0.001
    img_size = 256
    save_path = 'D:\\qy44lyfe\\LLM segmentation\Results\\2025\\Qwen 3_235B\\out of the box\\skin output'
    os.makedirs(save_path, exist_ok=True)

    # Dataset and DataLoader
    dataset = GrayscaleBinaryMaskDataset(image_dir=image_dir, mask_dir=mask_dir, mask_suffix=mask_suffix, img_size=img_size)
    indices = list(range(len(dataset)))

    # Split data
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)  # 10% of total

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Model Summary
    from torchinfo import summary
    summary(model, input_size=(1, 1, img_size, img_size))

    # Training Loop
    train_losses = []
    val_losses = []
    val_dice_all = []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_all.append(val_dice)

        # Save losses
        pd.DataFrame([list(range(1, epoch+2)), train_losses]).to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False, header=False)
        pd.DataFrame([list(range(1, epoch+2)), val_losses]).to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False, header=False)
        pd.DataFrame(val_dice_all).to_excel(os.path.join(save_path, 'validation_dice_scores.xlsx'), index=False)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state.pth'))
    torch.save(model, os.path.join(save_path, 'model_full.pth'))

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Test
    test_dice = test(model, test_loader, device, save_path)
    pd.DataFrame([test_dice]).to_excel(os.path.join(save_path, 'test_dice_scores.xlsx'), index=False)