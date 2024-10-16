# main.py
import torch
from torch import nn  # Add this import to include the neural network module
from torch.utils.data import DataLoader
from dataset import CustomDataset, get_transform
from model import UNet
from train import train_one_epoch, validate, save_losses, save_model, \
    save_dice_scores, plot_losses, visualize_predictions, test
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import time
import os
from torchinfo import summary

if __name__ == "__main__":
    data_folder = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output"
    os.makedirs(save_path, exist_ok=True)

    transform = get_transform()
    dataset = CustomDataset(data_folder, transform)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Print sizes
    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    summary(model, input_size=(16, 1, 256, 256))  # 1 channel for grayscale

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    num_epochs = 25
    train_losses, val_losses = [], []
    dice_scores_val = []
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device, save_path)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores_val.append(val_dice)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    save_losses(train_losses, val_losses, save_path)
    save_model(model, save_path)

    # save_dice_scores(dice_scores_val, save_path, 'validation_dice_scores')

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    dice_score_test = test(model, test_loader, loss_fn, device, save_path)
    print(f"Test Dice Score: {dice_score_test:.4f}")
    # save_dice_scores([dice_score_test], save_path, 'test_dice_scores')

    # Visualize predictions
    visualize_predictions(model, test_loader, device, save_path)
