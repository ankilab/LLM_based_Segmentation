# unet_segmentation/main.py

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from dataset import GrayscaleBinarySegmentationDataset
from model import UNet, print_model_summary
from train import train_one_epoch, validate, test, plot_losses

def main():
    # Hyperparameters
    data_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"
    save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\Qwen 3_235B_without thinking\\out of the box\\Brain output"
    os.makedirs(save_path, exist_ok=True)

    batch_size = 16
    epochs = 30
    lr = 1e-4
    img_size = (256, 256)
    num_workers = 4
    pin_memory = True

    # Dataset
    dataset = GrayscaleBinarySegmentationDataset(data_dir, mask_dir)

    # Split indices
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    print_model_summary(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop
    train_losses = []
    val_losses = []
    val_dice_scores_all = []

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice_scores = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores_all.append(val_dice_scores)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save losses
        pd.DataFrame([list(range(1, epoch+2)), train_losses]).T.to_excel(
            os.path.join(save_path, "train_losses.xlsx"), index=False, header=False)
        pd.DataFrame([list(range(1, epoch+2)), val_losses]).T.to_excel(
            os.path.join(save_path, "val_losses.xlsx"), index=False, header=False)

        # Save Dice scores per batch
        df_val_dice = pd.DataFrame(val_dice_scores_all)
        df_val_dice.to_excel(os.path.join(save_path, "validation_dice_scores.xlsx"))

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f}s")

    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, "model_state.pth"))
    torch.save(model, os.path.join(save_path, "model_full.pth"))

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Run testing
    test(model, test_loader, device, save_path)

if __name__ == "__main__":
    main()