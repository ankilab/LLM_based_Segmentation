# unet_segmentation/main.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch import optim
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import train_one_epoch, validate, test_model

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-3"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"Train loader: {len(train_loader)}, Val loader: {len(val_loader)}, Test loader: {len(test_loader)}")

    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))
    print(f"Total Learnable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()

    train_losses = []
    val_losses = []
    epochs = 20

    import time
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device, save_path, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} seconds")

    # Save losses
    pd.DataFrame([list(range(1, epochs+1)), train_losses]).to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False, header=False)
    pd.DataFrame([list(range(1, epochs+1)), val_losses]).to_excel(os.path.join(save_path, 'val_losses.xlsx'), index=False, header=False)

    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_state_dict.pth'))
    torch.save(model, os.path.join(save_path, 'unet_model.pth'))

    # Save loss plot
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))

    # Test and visualize
    test_model(model, test_loader, device, save_path)
