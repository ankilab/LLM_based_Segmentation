import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import time
from dataset import SegmentationDataset
from model import UNet
from train import train_fn, validate_fn, test_fn, save_losses, plot_losses, save_model, save_dice_scores, \
    visualize_predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs and Hyperparameters
    image_dir = r"D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    save_path = r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o"
    batch_size = 8
    learning_rate = 1e-5
    num_epochs = 20

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Loading Dataset
    dataset = SegmentationDataset(image_dir, transform=transform)
    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Training size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    # Model, Optimizer, Loss Function
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

    # Training Loop
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, device, epoch)
        val_loss, dice_scores = validate_fn(val_loader, model, loss_fn, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_dice_scores(dice_scores, save_path, "validation_dice_scores")

    # Save training progress
    save_losses(train_losses, val_losses, save_path)
    plot_losses(train_losses, val_losses, save_path)
    save_model(model, save_path)

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Testing the model
    test_dice_scores, visual_data = test_fn(test_loader, model, device)
    save_dice_scores(test_dice_scores, save_path, "test_dice_scores")
    visualize_predictions(visual_data, save_path)


if __name__ == "__main__":
    main()
