# unet_segmentation/main.py

import os
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import train_model, test_model

if __name__ == "__main__":
    # Paths
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-5"
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    image_size = (256, 256)
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 20

    # Prepare file list
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    all_files.sort()
    random.shuffle(all_files)

    # Split dataset
    train_idx, test_idx = train_test_split(range(len(all_files)), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    dataset = SegmentationDataset(image_dir, mask_dir, all_files, image_size=image_size)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet().to(device)
    summary(model, input_size=(1, 1, *image_size))
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss and Optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path)

    # Testing
    test_model(model, test_loader, device, save_path)
