import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os

from dataset import SegmentationDataset
from model import UNet
from train import train_model, test_model

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8
    num_epochs = 3
    learning_rate = 1e-4
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'  # Update this path
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\GPT 4o'  # Path to save results, ensure this directory exists

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(image_dir, target_size=(256, 256))  # Adjust target_size if needed
    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model, loss function, optimizer
    model = UNet()
    criterion = torch.nn.BCELoss()  # Since it's a binary segmentation problem
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, save_path)

    # Test the model
    test_model(model, test_loader, device, save_path)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model.pth"))
