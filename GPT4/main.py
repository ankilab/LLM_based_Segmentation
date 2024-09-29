# main.py
import torch
from torch import nn  # Add this import to include the neural network module
from torch.utils.data import DataLoader
from dataset import CustomDataset, get_transform
from model import UNet
from train import train_one_epoch, validate, save_losses, save_model
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    data_folder = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\GPT 4"
    os.makedirs(save_path, exist_ok=True)

    transform = get_transform()
    dataset = CustomDataset(data_folder, transform)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    num_epochs = 25
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    save_losses(train_losses, val_losses, save_path)
    save_model(model, save_path)
