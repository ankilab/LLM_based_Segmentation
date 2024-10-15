import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from model import UNet
from train import train, validate, test, save_losses, save_dice_scores, plot_losses, visualize_predictions
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T  # Ensure torchvision.transforms is imported
import time
from torchinfo import summary  # Import torchinfo

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8
    num_epochs = 25
    learning_rate = 1e-4
    image_size = (256, 256)  # Define a consistent image size for all inputs
    data_dir = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"
    save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\GPT 4o\\test"
    os.makedirs(save_path, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform to convert images to tensors
    transform = T.Compose([T.ToTensor()])

    # Load dataset
    dataset = SegmentationDataset(image_dir=data_dir, transform=transform, image_size=image_size)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    print(f'Training size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model
    model = UNet().to(device)
    # Print model summary using torchinfo
    summary(model, input_size=(batch_size, 1, image_size[0], image_size[1])) # 1 channel for grayscale

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation
    train_losses = []
    val_losses = []
    dice_scores_val = []

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, save_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores_val.append(val_dice)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save model
    torch.save(model, os.path.join(save_path, 'unet_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model_state_dict.pth'))

    # Save training losses and validation losses
    save_losses(train_losses, val_losses, save_path)
    # save_dice_scores(dice_scores_val, save_path, 'validation_dice_scores')

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    dice_score_test = test(model, test_loader, device, save_path)
    print(f"Test Dice Score: {dice_score_test:.4f}")
    # save_dice_scores([dice_score_test], save_path, 'test_dice_scores')

    # Visualize predictions
    visualize_predictions(model, test_loader, device, save_path)
