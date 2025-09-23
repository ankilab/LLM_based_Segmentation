# unet_segmentation/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
from torchinfo import summary
from dataset import SegmentationDataset
from model import UNet
from train import train, validate, test, plot_losses

if __name__ == "__main__":
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"  # or same as image_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-8"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    indices = list(range(len(dataset)))
    train_idx, testval_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    print(f"Train size: {len(train_idx)}")
    print(f"Val size: {len(val_idx)}")
    print(f"Test size: {len(test_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    num_epochs = 20
    train_losses, val_losses = [], []

    import time
    start = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    duration = time.time() - start
    print(f"Training completed in {duration:.2f} seconds")

    pd.DataFrame([range(1, num_epochs + 1), train_losses]).to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False, header=False)
    pd.DataFrame([range(1, num_epochs + 1), val_losses]).to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False, header=False)

    torch.save(model, os.path.join(save_path, "full_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "model_weights.pth"))

    test(model, test_loader, device, save_path)
    plot_losses(train_losses, val_losses, save_path)
