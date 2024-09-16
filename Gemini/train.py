import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import ToTensor, Normalize

from Dataset import SegmentationDataset
from model import Unet


def train(model, train_loader, val_loader, learning_rate, num_epochs, device):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss: {epoch_loss:.4f}")

        validate(model, val_loader, device)

    torch.save(model.state_dict(), "unet_model.pth")


def validate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(val_loader)
        print(f"Validation Loss: {epoch_loss:.4f}")