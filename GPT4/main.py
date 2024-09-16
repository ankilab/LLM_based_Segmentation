import torch
import torch.nn as nn
from train import get_dataloaders, train_loop, test_loop
from model import UNet

def main():
    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'
    batch_size = 4
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_dataloaders(image_dir, mask_dir, batch_size)
    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loop(val_loader, model, loss_fn, device)
    test_loop(test_loader, model, loss_fn, device)

if __name__ == "__main__":
    main()
