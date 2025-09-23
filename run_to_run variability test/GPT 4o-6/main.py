# unet_segmentation/main.py

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchinfo import summary
import os

from dataset import GraySegmentationDataset
from model import UNet
from train import *

if __name__ == "__main__":
    # === Config ===
    image_path = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_path = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-6"
    os.makedirs(save_path, exist_ok=True)

    batch_size = 8
    lr = 1e-4
    epochs = 2 #25
    img_size = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([T.Resize(img_size), T.ToTensor()])

    # === Dataset & Splits ===
    full_dataset = GraySegmentationDataset(image_path, mask_path, transform=transform)
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Val Loader: {len(val_loader)} batches")
    print(f"Test Loader: {len(test_loader)} batches")

    # === Model ===
    model = UNet().to(device)
    summary(model, input_size=(1, 1, *img_size))

    # === Optimizer, Loss ===
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses, val_losses = [], []
    val_dice_all = []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_all.append(val_dice)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {sum(val_dice)/len(val_dice):.4f}")

    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")

    # Save model
    torch.save(model, os.path.join(save_path, "model_full.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "model_state_dict.pth"))

    # Save logs
    save_loss_excel(train_losses, os.path.join(save_path, "train_losses.xlsx"))
    save_loss_excel(val_losses, os.path.join(save_path, "val_losses.xlsx"))
    save_dice_scores_excel(val_dice_all, os.path.join(save_path, "validation_dice_scores.xlsx"))
    visualize_losses(train_losses, val_losses, os.path.join(save_path, "loss_plot.png"))

    # Test & Save test dice
    test_dice = test(model, test_loader, device)
    save_dice_scores_excel([test_dice], os.path.join(save_path, "test_dice_scores.xlsx"))

    # Visualize prediction
    visualize_predictions(model, full_dataset, device, os.path.join(save_path, "sample_predictions.png"))
