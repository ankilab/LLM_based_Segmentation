import torch
import torch.nn as nn
import torch.optim as optim
from train import train, validate, test, load_data
from model import UNet

'''
Please provide a complete ready-to-run Python implementation for a U-Net architecture using PyTorch for binary segmentation of grayscale images. The dataset should consist of grayscale images and binary masks. The code should be structured into importable scripts with the following:

Required library imports.
A custom Dataset class for loading the grayscale images and corresponding binary masks.
A UNet model class definition using PyTorch, optimized for binary segmentation.
Functions for the training procedure, validation procedure, and testing procedure.
A function to split the dataset into 80% training, 10% validation, and 10% testing sets using train_test_split.
DataLoader setup for training, validation, and testing sets.
A function to visualize the predicted masks, the input image, and the ground truth during testing.
All the above functionalities should be implemented in a modular way so that they can be imported as scripts or classes.
Ensure that everything can be executed in an if __name__ == "__main__": block with instructions for loading the dataset, configuring hyperparameters, and saving the trained model.. the directory structure should be: unet_segmentation/
├── dataset.py
├── model.py
├── train.py
├── main.py

'''

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    img_dir = 'path_to_images'
    mask_dir = 'path_to_masks'
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 16

    # Load data
    train_loader, val_loader, test_loader = load_data(img_dir, mask_dir, batch_size)

    # Initialize model, loss function, and optimizer
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Test the model
    test_predictions = test(model, test_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'unet_model.pth')
