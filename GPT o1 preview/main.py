# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CustomSegmentationDataset
from model import UNet
from train import train_model, validate_model, test_model, save_losses_to_excel, plot_losses, visualize_predictions
# from torchsummary import summary
from torchinfo import summary
import time
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    # Paths
    dataset_path = 'D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset'
    image_dir = dataset_path
    mask_dir = dataset_path

    # Save path for outputs
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get list of image files
    images = [img for img in os.listdir(image_dir) if img.endswith('.png') and not img.endswith('_seg.png')]
    images.sort()

    # Shuffle and split the data
    train_imgs, val_test_imgs = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)
    val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=42, shuffle=True)

    # Print sizes
    print(f"Total images: {len(images)}")
    print(f"Training images: {len(train_imgs)}")
    print(f"Validation images: {len(val_imgs)}")
    print(f"Test images: {len(test_imgs)}")

    # Create directories for splits
    split_dirs = ['train', 'val', 'test']
    for split_dir in split_dirs:
        img_split_dir = os.path.join(save_path, split_dir, 'images')
        mask_split_dir = os.path.join(save_path, split_dir, 'masks')
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(mask_split_dir, exist_ok=True)

    # Copy files
    def copy_files(file_list, split):
        img_dest = os.path.join(save_path, split, 'images')
        mask_dest = os.path.join(save_path, split, 'masks')
        for img_name in tqdm(file_list, desc=f"Copying {split} data"):
            mask_name = img_name.replace('.png', '_seg.png')
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(img_dest, img_name))
            shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(mask_dest, mask_name))

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')

    # Hyperparameters
    image_size = (256, 256)
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4

    # Datasets
    train_dataset = CustomSegmentationDataset(
        image_dir=os.path.join(save_path, 'train', 'images'),
        mask_dir=os.path.join(save_path, 'train', 'masks'),
        image_size=image_size
    )

    val_dataset = CustomSegmentationDataset(
        image_dir=os.path.join(save_path, 'val', 'images'),
        mask_dir=os.path.join(save_path, 'val', 'masks'),
        image_size=image_size
    )

    test_dataset = CustomSegmentationDataset(
        image_dir=os.path.join(save_path, 'test', 'images'),
        mask_dir=os.path.join(save_path, 'test', 'masks'),
        image_size=image_size
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train DataLoader size: {len(train_loader)}")
    print(f"Validation DataLoader size: {len(val_loader)}")
    print(f"Test DataLoader size: {len(test_loader)}")

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary and total parameters
    # summary(model, input_size=(1, image_size[0], image_size[1]))
    summary(model, input_size=(batch_size, 1, image_size[0], image_size[1]))  # 1 channel for grayscale
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Training
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device, epoch, save_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save the model
    torch.save(model, os.path.join(save_path, 'unet_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model_state_dict.pth'))

    # Save losses to Excel files
    save_losses_to_excel(train_losses, val_losses, save_path)

    # Plot losses
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    all_inputs, all_masks, all_preds, all_img_names = test_model(model, test_loader, device, save_path)

    # Visualize predictions
    visualize_predictions(all_inputs, all_masks, all_preds, all_img_names, save_path)
