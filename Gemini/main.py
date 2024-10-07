import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train
from dataset import SegmentationDataset
from model import Unet

if __name__ == "__main__":
    data_dir = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Gemini"

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust image size if needed
        transforms.ToTensor(),
    ])

    # Create datasets
    dataset = SegmentationDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Create U-Net model
    model = Unet(in_channels=1, out_channels=1)

    # Print model summary and number of parameters
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Train the model
    train(model, train_loader, val_loader, num_epochs=2, learning_rate=0.001, save_path=save_path)