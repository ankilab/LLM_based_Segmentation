# unet_segmentation/main.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from model import UNet
from train import train, test
from torchinfo import summary

if __name__ == "__main__":
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"  # Path to images
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"   # Same directory
    save_dir = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-1"
    batch_size = 8
    epochs = 30
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)
    indices = list(range(len(full_dataset)))
    train_idx, testval_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, random_state=42)

    print(f"Total: {len(full_dataset)}, Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1, shuffle=False)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    model = UNet().to(device)
    summary(model, input_size=(1, 1, 256, 256), device=str(device))

    import time
    start_time = time.time()
    train(model, train_loader, val_loader, device, save_dir, epochs, lr)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")

    test(model, test_loader, device, save_dir)
