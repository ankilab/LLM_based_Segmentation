import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import GrayscaleSegmentationDataset
from model import UNet

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def test(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu())
    return predictions

def load_data(img_dir, mask_dir, batch_size=16, test_split=0.2, val_split=0.1):
    dataset = GrayscaleSegmentationDataset(img_dir, mask_dir, transform=None)
    train_data, test_data = train_test_split(dataset, test_size=test_split, random_state=42)
    val_size = int(val_split * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
