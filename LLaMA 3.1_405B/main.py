import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from dataset import BinarySegmentationDataset
from model import UNet
from train import train, validate, test, save_losses, visualize_losses, visualize_predictions
import time

def main():
    # Set hyperparameters
    batch_size = 32
    #epochs = 10
    epochs = 2
    learning_rate = 0.001
    # data_folder = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    # data_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Swallowing\images"
    data_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"

    # mask_folder = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"
    # mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Swallowing\masks"
    mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"


    #save_path = 'D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output'
    #save_path = 'D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\Bolus output'
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\Brain output'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = BinarySegmentationDataset(data_folder, mask_folder)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Print sizes
    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and criterion
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    model.to(device)
    # Print model summary using torchinfo
    summary(model, input_size=(batch_size, 1, 256, 256))  # 1 channel for grayscale

    # Train model
    train_losses = []
    val_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, dice_scores = validate(model, device, val_loader, criterion, epoch, save_path)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save model and losses
    # torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    torch.save(model, os.path.join(save_path, 'unet_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model_state_dict.pth'))
    save_losses(train_losses, val_losses, save_path)

    # Visualize losses
    visualize_losses(train_losses, val_losses, save_path)

    # Test model
    test_dice_scores = test(model, device, test_loader, save_path)

    # Visualize predictions
    visualize_predictions(model, test_loader, device, save_path)

if __name__ == '__main__':
    main()