# unet_segmentation/main.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchinfo import summary
import numpy as np

from dataset import SegmentationDataset
from model import UNet
from train import train, validate, test, plot_losses, save_excel

if __name__ == "__main__":
    # === CONFIGURATION ===
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or same as image_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-7"
    os.makedirs(save_path, exist_ok=True)

    epochs = 25
    batch_size = 8
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)

    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    test_data = Subset(dataset, test_idx)

    print(f"Dataset Sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print(f"DataLoader Sizes: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    import time
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    duration = time.time() - start_time
    print(f"\nTotal training time: {duration:.2f} seconds")

    save_excel([list(range(1, epochs + 1)), train_losses], os.path.join(save_path, "train_losses.xlsx"))
    save_excel([list(range(1, epochs + 1)), val_losses], os.path.join(save_path, "val_losses.xlsx"))

    torch.save(model, os.path.join(save_path, "unet_model.pt"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_state_dict.pth"))

    test(model, test_loader, device, save_path)
    plot_losses(train_losses, val_losses, save_path)
