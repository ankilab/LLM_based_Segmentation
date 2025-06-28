import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from model import UNet
from train import train_model, test_model, visualize_predictions
from torchinfo import summary

def main():
    # Paths and hyperparameters
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_images\converted'  # Replace with your image directory
    # mask_dir = None  # Set to None if masks are in the same directory with a suffix
    mask_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_masks\converted'
    mask_suffix = ''
    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Mistral Medium 3\\out of the box\\Retina output'  # Directory to save results
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    image_size = (256, 256)  # Resize images to this size

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = SegmentationDataset(image_dir, mask_dir, mask_suffix, transform=transform)
    #dataset = SegmentationDataset(image_dir, mask_suffix="_seg", transform=transform)

    # Split dataset into train, validation, and test sets
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=42)  # 0.125 x 0.8 = 0.1

    print(f"Train size: {len(train_indices)}")
    print(f"Validation size: {len(val_indices)}")
    print(f"Test size: {len(test_indices)}")

    # Create data loaders
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Train loader size: {len(train_loader)}")
    print(f"Validation loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")

    # Initialize model
    model = UNet(n_channels=1, n_classes=1)

    # Print model summary
    #summary(model, input_size=(1, *image_size))
    summary(model, input_size=(batch_size, 1, *image_size))

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model, train_losses, val_losses, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path
    )

    # Test the model
    test_loss, test_dice_scores = test_model(trained_model, test_loader, criterion, save_path)

    # Visualize predictions
    visualize_predictions(trained_model, test_loader, save_path)

if __name__ == "__main__":
    main()