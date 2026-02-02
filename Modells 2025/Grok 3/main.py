import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchinfo import summary
import time
from dataset import SegmentationDataset
from model import UNet
from train import train_model, validate_model, test_model, save_losses, plot_losses, visualize_predictions

def main():
    # Hyperparameters and configurations
    IMAGE_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"  # Update with your dataset path
    MASK_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"    # Update if masks are in a different folder
    SAVE_PATH = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Grok 3\\out of the box\\BAGLS output"
    os.makedirs(SAVE_PATH, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 256

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Dataset and splits
    dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transform=transform)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, train_size=0.8, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # Model setup
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    summary(model, input_size=(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    start_time = time.time()
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_dice = validate_model(model, val_loader, criterion, DEVICE, epoch, SAVE_PATH)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    # Save losses and plot
    save_losses(train_losses, val_losses, SAVE_PATH)
    plot_losses(train_losses, val_losses, SAVE_PATH)

    # Save model
    torch.save(model, os.path.join(SAVE_PATH, "full_model.pth"))
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_state_dict.pth"))

    # Testing
    predictions, ground_truths, input_images, filenames = test_model(model, test_loader, DEVICE, SAVE_PATH)
    visualize_predictions(predictions, ground_truths, input_images, filenames, SAVE_PATH)

if __name__ == "__main__":
    main()