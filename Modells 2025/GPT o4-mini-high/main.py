import os
import time
import argparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

from dataset import GrayscaleDataset
from model import UNet
from train import (
    train_one_epoch,
    validate,
    test,
    save_losses_to_excel,
    save_dice_scores_to_excel,
    plot_losses,
    visualize_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for binary segmentation (grayscale).")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Path to folder with input grayscale images (PNG).")
    parser.add_argument('--mask_dir', type=str, required=True,
                        help="Path to folder with binary mask PNGs (named <basename>_seg.png).")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory where models, logs, and plots will be saved.")
    parser.add_argument('--image_size', type=int, default=256,
                        help="Resize images and masks to this size (square).")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create save directory if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Prepare dataset (do not copy files; split via indices)
    full_dataset = GrayscaleDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size
    )

    n_total = len(full_dataset)
    indices = list(range(n_total))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)

    print(f"Total samples: {n_total}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Testing samples: {len(test_idx)}")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    # 2) DataLoaders
    train_loader = DataLoader(train_subset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_subset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_subset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # 3) Initialize model, optimizer, loss
    device = torch.device(args.device)
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Print model summary and total parameters
    print("\nModel Summary:")
    summary(model, input_size=(args.batch_size, 1, args.image_size, args.image_size))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4) Training loop
    train_losses = []
    val_losses = []
    all_val_dice = []  # list of lists (epochs × batches)

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)

        # Validation
        val_loss, val_batch_dice = validate(model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)
        all_val_dice.append(val_batch_dice)

        print(f"Epoch {epoch}/{args.num_epochs} → Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Dice (last batch): {val_batch_dice[-1]:.4f}")

    end_time = time.time()
    elapsed = end_time - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTotal training time: {int(hrs)}h {int(mins)}m {int(secs)}s")

    # 5) Save training/validation losses to Excel
    train_losses_path = os.path.join(args.save_dir, 'train_losses.xlsx')
    val_losses_path = os.path.join(args.save_dir, 'val_losses.xlsx')
    save_losses_to_excel(train_losses, train_losses_path)
    save_losses_to_excel(val_losses, val_losses_path)

    # 6) Save validation dice scores to Excel (epochs × batches)
    val_dice_path = os.path.join(args.save_dir, 'validation_dice_scores.xlsx')
    save_dice_scores_to_excel(all_val_dice, val_dice_path)

    # 7) Save model (entire and state_dict)
    model_full_path = os.path.join(args.save_dir, 'unet_model.pth')
    model_state_dict_path = os.path.join(args.save_dir, 'unet_model_state_dict.pth')
    torch.save(model, model_full_path)
    torch.save(model.state_dict(), model_state_dict_path)

    # 8) Plot training vs validation loss
    loss_plot_path = os.path.join(args.save_dir, 'loss_curves.png')
    plot_losses(train_losses, val_losses, loss_plot_path)
    print(f"Saved loss curves to {loss_plot_path}")

    # 9) Testing
    test_loss, test_batch_dice = test(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    # Save test dice scores (only one "epoch" so one row, multiple batches)
    test_dice_path = os.path.join(args.save_dir, 'test_dice_scores.xlsx')
    # Reshape to 1 × num_batches
    save_dice_scores_to_excel([test_batch_dice], test_dice_path)

    # 10) Visualize 5 random samples from test set
    vis_path = os.path.join(args.save_dir, 'test_predictions.png')
    visualize_predictions(model, test_subset.dataset, device, vis_path, num_samples=5)
    print(f"Saved sample predictions to {vis_path}")
