# main.py

import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from Dataset import GrayscaleSegmentationDataset
from model import UNet
from train import train_model, validate_model, test_model
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(images, masks, predictions, num_samples=5):
    for i in range(num_samples):
        image = images[i].squeeze().numpy()
        mask = masks[i].squeeze().numpy()
        prediction = predictions[i].squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Input Image')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(prediction, cmap='gray')
        axs[2].set_title('Predicted Mask')
        for ax in axs:
            ax.axis('off')
        plt.show()

if __name__ == "__main__":
    # Hyperparameters
    images_dir = 'path/to/images'  # Replace with your images directory
    masks_dir = 'path/to/masks'    # Replace with your masks directory
    model_save_path = 'unet_model.pth'
    num_epochs = 25
    batch_size = 8
    learning_rate = 1e-3

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Dataset
    dataset = GrayscaleSegmentationDataset(images_dir, masks_dir, transform=transform)

    # Split dataset into training, validation, and testing
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices, val_test_indices = train_test_split(indices, test_size=(val_size + test_size), random_state=42)
    val_indices, test_indices = train_test_split(val_test_indices, test_size=(test_size / (val_size + test_size)), random_state=42)

    # Subset the datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print('Model saved.')

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))

    # Testing
    images_list, masks_list, predictions = test_model(model, test_loader, device)

    # Convert lists to tensors
    images_concat = torch.cat(images_list, dim=0)
    masks_concat = torch.cat(masks_list, dim=0)
    preds_concat = torch.cat(predictions, dim=0)
    preds_concat = (preds_concat > 0.5).float()  # Threshold predictions

    # Visualize results
    visualize_results(images_concat, masks_concat, preds_concat)
