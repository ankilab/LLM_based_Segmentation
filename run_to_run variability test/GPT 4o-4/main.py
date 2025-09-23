# unet_segmentation/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import *
from torchinfo import summary

if __name__ == "__main__":
    # Paths
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or same as image_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-4"
    os.makedirs(save_path, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = GrayscaleSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    indices = list(range(len(dataset)))
    train_idx, testval_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=1)

    print(f"Train loader: {len(train_loader)} | Val loader: {len(val_loader)} | Test loader: {len(test_loader)}")

    # Model
    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    epochs = 20

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device, save_path, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    duration = time.time() - start_time
    print(f"Total training time: {duration:.2f} seconds")

    # Save model
    torch.save(model, os.path.join(save_path, "model_full.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "model_state_dict.pth"))

    # Save losses
    pd.DataFrame([list(range(1, epochs + 1)), train_losses]).to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False, header=False)
    pd.DataFrame([list(range(1, epochs + 1)), val_losses]).to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False, header=False)

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Test
    predictions = test(model, test_loader, device, save_path)
    visualize_predictions(predictions, save_path)
