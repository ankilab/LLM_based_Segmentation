# unet_segmentation/main.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchinfo import summary
from dataset import SegmentationDataset
from model import UNet
from train import train_epoch, validate_epoch, test, plot_losses
import pandas as pd

if __name__ == "__main__":
    # Setup
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
    mask_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-10'
    os.makedirs(save_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 25
    train_losses = []
    val_losses = []

    import time
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

    # Save losses
    pd.DataFrame([list(range(1, num_epochs+1)), train_losses]).to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False, header=False)
    pd.DataFrame([list(range(1, num_epochs+1)), val_losses]).to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False, header=False)

    # Save model
    torch.save(model, os.path.join(save_path, "model_full.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "model_state.pth"))

    # Plot
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    test(model, test_loader, device, save_path)
