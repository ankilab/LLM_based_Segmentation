# main.py
import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchinfo import summary

from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import train_one_epoch, validate, test

import matplotlib.pyplot as plt
import time

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))

if __name__ == "__main__":
    # Configuration
    set_seed(42)
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or same as image_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-6"
    os.makedirs(save_path, exist_ok=True)

    batch_size = 8
    lr = 1e-4
    epochs = 2 #25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = GrayscaleSegmentationDataset(image_dir, mask_dir)
    idxs = list(range(len(dataset)))
    train_idx, testval_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(f"Dataset sizes -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print(f"Loader sizes -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Model
    model = UNet()
    model.to(device)
    summary(model, input_size=(1, 1, 256, 256))
    print("Total params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device, save_path, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save losses after each epoch
        pd.DataFrame([range(1, len(train_losses)+1), train_losses]).to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False, header=False)
        pd.DataFrame([range(1, len(val_losses)+1), val_losses]).to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False, header=False)

    # Save model
    torch.save(model, os.path.join(save_path, "unet_model.pt"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_state_dict.pth"))

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Final testing
    test(model, test_loader, device, save_path)
