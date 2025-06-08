# unet_segmentation/main.py

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchinfo import summary

from dataset import UNetDataset
from model import UNet
from train import (
    DiceLoss,
    train_epoch,
    validate_epoch,
    test_model,
    save_losses,
    save_dice_scores,
    plot_and_save_losses,
    visualize_predictions
)


def main():
    # --- Configuration ---
    # Path to your data
    IMAGE_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    MASK_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    SAVE_PATH = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Gemini 2.5 Pro\\out of the box\\BAGLS output"

    # Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    #NUM_EPOCHS = 25  # Adjust as needed
    NUM_EPOCHS = 3  # Adjust as needed
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    MASK_SUFFIX = "_seg"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])

    # Get all valid image filenames (those ending in .png)
    all_image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])

    # Create indices and split them
    indices = list(range(len(all_image_files)))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=1 / 9,
                                                  random_state=42)  # 1/9 of 90% is 10% of total

    # Create the full dataset instance
    full_dataset = UNetDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_filenames=all_image_files,
        mask_suffix=MASK_SUFFIX,
        transform=image_transform,
        mask_transform=mask_transform
    )

    # Create subsets for training, validation, and testing
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Total samples: {len(all_image_files)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # --- Model, Loss, and Optimizer ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    print("\n--- Model Summary ---")
    summary(model, input_size=(BATCH_SIZE, 1, IMG_HEIGHT, IMG_WIDTH))

    # Loss: Using BCEWithLogitsLoss for numerical stability + custom DiceLoss
    loss_fn_bce = torch.nn.BCEWithLogitsLoss()
    loss_fn_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()

    all_train_losses = []
    all_val_losses = []
    all_val_dice_scores = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_epoch(train_loader, model, optimizer, loss_fn_bce, loss_fn_dice, device)
        val_loss, val_dice_scores = validate_epoch(val_loader, model, loss_fn_bce, loss_fn_dice, device)

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_val_dice_scores.append(val_dice_scores)

        print(f"Epoch {epoch + 1} Summary: Avg Train Loss: {train_loss:.4f}, Avg Val Loss: {val_loss:.4f}")

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f} seconds")

    # --- Saving Results ---
    print("\n--- Saving Results ---")
    # Save losses to Excel and plot them
    save_losses(all_train_losses, all_val_losses, SAVE_PATH)
    plot_and_save_losses(SAVE_PATH)

    # Save validation dice scores
    save_dice_scores(all_val_dice_scores, SAVE_PATH, "validation_dice_scores.xlsx")

    # Save the model
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'unet_model_state_dict.pth'))
    torch.save(model, os.path.join(SAVE_PATH, 'unet_model_full.pth'))
    print("Models, losses, and plots saved successfully.")

    # --- Testing ---
    print("\n--- Starting Testing ---")
    test_dice_scores = test_model(test_loader, model, device)
    save_dice_scores([test_dice_scores], SAVE_PATH, "test_dice_scores.xlsx")
    print(f"Average Test Dice Score: {sum(test_dice_scores) / len(test_dice_scores):.4f}")

    # --- Visualization ---
    print("\n--- Visualizing Test Predictions ---")
    visualize_predictions(model, test_dataset, device, SAVE_PATH, num_samples=5)
    print("Test prediction visualizations saved.")

    print("\n--- Process Complete ---")


if __name__ == '__main__':
    main()