# main.py
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from model import UNet
from train import train_model, test_model, plot_losses
from torchinfo import summary

if __name__ == "__main__":
    # Config
    data_path = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_path = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or same as data_path
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-2"
    os.makedirs(save_path, exist_ok=True)
    img_size = (256, 256)
    batch_size = 8
    epochs = 2 #25
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = SegmentationDataset(data_path, mask_path, transform_size=img_size)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} | Test loader: {len(test_loader)}")

    # Model
    model = UNet(in_channels=1, out_channels=1)
    summary(model, input_size=(1, 1, img_size[0], img_size[1]))
    print(f"Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, save_path, epochs=epochs, lr=lr)

    # Test
    test_model(model, test_loader, device, save_path)

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)
