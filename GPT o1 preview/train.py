# train.py

import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test_model(model, dataloader, device):
    model.eval()
    predictions = []
    images_list = []
    masks_list = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            predictions.append(outputs.cpu())
            images_list.append(images.cpu())
            masks_list.append(masks.cpu())

    return images_list, masks_list, predictions
