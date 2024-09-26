import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SegmentationDataset
from model import UNet
from train import train_model, test_model, visualize_predictions
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Parameters
    root_dir = r"D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    save_path = r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o"
    # epochs = 50
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SegmentationDataset(root_dir, transform=transform)
    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    # Train model
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_path, device)

    # Test model
    test_model(model, test_loader, save_path, device)

    # Visualize predictions
    visualize_predictions(model, test_loader, save_path, device)
