import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchinfo
from dataset import GrayscaleSegmentationDataset, get_transforms
from model import get_unet_model
from train import train_model, test_model
import os

if __name__ == "__main__":
    # Configuration
    root_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'  # Update this
    save_path = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3 mini Reasoning\\out of the box\\BAGLS output'  # Where to save outputs
    os.makedirs(save_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16  # Chosen to avoid overfitting
    learning_rate = 1e-4  # Small LR
    num_epochs = 20

    # Create dataset
    dataset = GrayscaleSegmentationDataset(root_dir, transform=get_transforms())
    print(f'Total samples: {len(dataset)}')

    # Split dataset (80% train, 10% val, 10% test) on indices
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)  # 10% of total

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    print(f'Train set size: {len(train_subset)}')
    print(f'Validation set size: {len(val_subset)}')
    print(f'Test set size: {len(test_subset)}')

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f'Train loader size: {len(train_loader.dataset)} samples')
    print(f'Val loader size: {len(val_loader.dataset)} samples')
    print(f'Test loader size: {len(test_loader.dataset)} samples')

    # Model, Optimizer, etc.
    model = get_unet_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary
    torchinfo.summary(model, input_size=(1, 1, 256, 256))  # Assuming 1 channel input
    print(f'Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Train
    train_model(model, train_loader, val_loader, optimizer, None, num_epochs, device, save_path)

    # Test
    test_model(model, test_loader, device, save_path)