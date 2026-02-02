import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
from tqdm import tqdm
from torchinfo import summary
from dataset import SegmentationDataset, get_transform
from model import UNet
from train import train, test, visualize_predictions


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    #data_path = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    data_path = "D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_images\converted"
    #data_path = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"

    # mask_folder = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"
    mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_masks\converted"
    #mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"

    #save_path = 'D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output'
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\Retina output'
    #save_path = 'D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\Brain output'

    os.makedirs(save_path, exist_ok=True)

    # Create dataset
    dataset = SegmentationDataset(data_path, mask_folder, transform=get_transform())

    # Split dataset
    train_val_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

    train_size = int(0.8 * len(dataset))
    val_size = train_val_size - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    print(f"Total images: {len(dataset)}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")

    # Create model
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Set up loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print model summary using torchinfo
    summary(model, input_size=(batch_size, 1, 256, 256))  # 1 channel for grayscale

    # Train the model
    num_epochs = 50
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)

    # Test the model
    best_model = UNet(n_channels=1, n_classes=1).to(device)
    best_model.load_state_dict(torch.load(f'{save_path}/best_model.pth'))
    test_dice_scores = test(best_model, test_loader, device, save_path)

    print(f"Average Dice score on test set: {np.mean(test_dice_scores):.4f}")

    # Visualize predictions
    visualize_predictions(best_model, test_dataset, device, save_path)


if __name__ == '__main__':
    main()