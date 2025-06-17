import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import GrayscaleDataset
from model import UNet
from train import train, validate, test, save_losses, save_dice_scores, plot_losses, visualize_predictions
import torchinfo

def main():
    # Hyperparameters
    batch_size = 32
    epochs = 10
    lr = 0.001
    image_path = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    mask_path = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Llama 4 Maverick\\out of the box\\BAGLS output'

    # Load dataset
    dataset = GrayscaleDataset(image_path, mask_path)
    indices = np.arange(len(dataset))
    train_idx, val_test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Testing samples: {len(test_dataset)}')

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    torchinfo.summary(model, input_size=(batch_size, 1, 256, 256))

    # Training
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    val_dice_scores = []
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss, dice_scores = validate(model, device, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(dice_scores)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')

    # Save model and losses
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    torch.save(model, os.path.join(save_path, 'model_full.pth'))
    save_losses(train_losses, val_losses, save_path)
    save_dice_scores(val_dice_scores, save_path, 'validation_dice_scores.xlsx')
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    test_loss, test_dice_scores = test(model, device, test_loader, criterion)
    save_dice_scores([test_dice_scores], save_path, 'test_dice_scores.xlsx')
    visualize_predictions(model, device, test_loader, save_path)

if __name__ == '__main__':
    main()