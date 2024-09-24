import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from model import UNet
from dataset import SegmentationDataset


def dice_score(pred, target):
    pred = torch.round(pred)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum())


def validate_model(model, criterion, val_loader, device):
    model.eval()
    dice_scores = []  # List to hold Dice scores for this epoch

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate Dice score and append to list
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())

    return dice_scores  # Return all dice scores for the current epoch


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, save_path):
    model.to(device)
    start_time = time.time()

    # Dictionary to store Dice scores per epoch
    dice_scores_dict = {}

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_loader = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # Validate the model and get Dice scores for this epoch
        dice_scores = validate_model(model, criterion, val_loader, device)

        # Store dice scores in dictionary (key: epoch number, value: list of Dice scores for batches)
        dice_scores_dict[epoch + 1] = dice_scores

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_train_loss / len(train_loader)}, '
              f'Val Dice: {sum(dice_scores) / len(dice_scores)}')

    duration = time.time() - start_time
    print(f"Training completed in {duration // 60:.0f}m {duration % 60:.0f}s")

    # Convert the dice_scores_dict to a DataFrame and save it to Excel
    save_dice_scores_to_excel(dice_scores_dict, save_path)


def save_dice_scores_to_excel(dice_scores_dict, save_path):
    # Convert the dictionary of lists to a DataFrame
    df = pd.DataFrame.from_dict(dice_scores_dict, orient='index')

    # Name the columns as 'Batch 1', 'Batch 2', etc.
    df.columns = [f'Batch {i + 1}' for i in range(df.shape[1])]

    # Name the rows as 'Epoch 1', 'Epoch 2', etc.
    df.index.name = 'Epoch'

    # Save the DataFrame to an Excel file
    excel_file_path = os.path.join(save_path, "validation_dice_scores.xlsx")
    df.to_excel(excel_file_path)

    print(f'Dice scores saved to {excel_file_path}')


def plot_losses(train_losses, val_losses, save_path):
    """Function to plot and save training/validation losses."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Save the plot as a .png file
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.show()


def test_model(model, test_loader, device, save_path):
    model.eval()
    dice_scores = []
    test_loader = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())

    # Save dice scores to Excel file
    df = pd.DataFrame({"Dice Score": dice_scores})
    df.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"), index=False)

    visualize_predictions(model, test_loader, device, save_path)


def visualize_predictions(model, test_loader, device, save_path):
    model.eval()
    images, masks = next(iter(test_loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)

    plt.figure(figsize=(15, 15))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title("Input Image")

        plt.subplot(3, 5, i + 6)
        plt.imshow(masks[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title("Ground Truth")

        plt.subplot(3, 5, i + 11)
        plt.imshow(outputs[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title("Prediction")

    plt.savefig(os.path.join(save_path, "predictions.png"))
    plt.show()
