import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset import SegmentationDataset
from model import UNet
import torch.nn as nn
import torch.optim as optim
from train import train_model, test_model

if __name__ == "__main__":
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\Copilot'
    os.makedirs(save_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(image_dir, transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 25
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, save_path)
    test_model(model, test_loader, device, save_path)