import torch
from torchvision import transforms

from Dataset import SegmentationDataset
from train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths
    image_dir = "path/to/your/images"
    mask_dir = "path/to/your/masks"

    # Transformations
    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = Unet().to(device)

    # Train the model
    train(model, train_loader, val_loader, learning_rate=0.001, num_epochs=10, device=device)


if __name__ == "__main__":
    main()