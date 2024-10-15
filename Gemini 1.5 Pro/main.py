import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pandas as pd
import shutil #for copying files
import time
from torchinfo import summary
import torch.optim as optim


from dataset import SegmentationDataset #importing from other scripts
from model import UNet
from train import train_one_epoch, validate, test, visualize_predictions, plot_losses

if __name__ == "__main__":
    # 0. Set Hyperparameters, paths, etc
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset" # Replace with the actual path to your dataset
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro" #path to save model, logs, etc
    os.makedirs(save_path, exist_ok=True)

    img_size = (256, 256) #resize images to this size
    batch_size = 8  # Adjust based on your GPU memory
    learning_rate = 1e-4
    num_epochs = 20  # Adjust as needed
    # num_epochs = 2


    # 1. Create Dataset and DataLoaders
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(image_dir, transform=transform)


    # 1. Split data
    train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, shuffle=True)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)


    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    #copy splitted data to new folders
    splits = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    for split_name, split_data in splits.items():
        split_dir = os.path.join(save_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for idx in range(len(split_data)):
            original_image_name = dataset.image_files[split_data.indices[idx]]
            original_mask_name = original_image_name.split('.')[0] + "_seg.png"
            shutil.copy(os.path.join(image_dir, original_image_name), split_dir)
            shutil.copy(os.path.join(image_dir, original_mask_name), split_dir)

    print(f"Total images: {len(dataset)}")
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train DataLoader size:", len(train_loader))
    print("Val DataLoader size:", len(val_loader))
    print("Test DataLoader size:", len(test_loader))


    # 2. Model, Optimizer, Loss, Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss() #binary cross entropy loss for binary segmentation

    #print model summary
    # summary(model, input_size=(1, img_size[0], img_size[1]))
    # Print model summary using torchinfo
    summary(model, input_size=(batch_size, 1, img_size[0], img_size[1]))  # 1 channel for grayscale
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")


    # 3. Training Loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    val_dice_all = [] #store all validation dice scores for all epochs and batches
    test_dice_all = [] #store all test dice scores for all epochs and batches

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch + 1, save_path)
        # val_dice_all.append(val_dice.cpu().item())
        # val_losses.append(val_loss.cpu().item())
        val_losses.append(val_loss)
        #val_dice_all.append(val_dice)

        # Save losses to Excel
        df_train = pd.DataFrame({'Epoch': range(1, epoch + 2), 'Train Loss': train_losses})
        df_train.to_excel(os.path.join(save_path, "train_losses.xlsx"), index=False)

        df_val = pd.DataFrame({'Epoch': range(1, epoch + 2), 'Val Loss': val_losses})
        df_val.to_excel(os.path.join(save_path, "val_losses.xlsx"), index=False)


    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")


    # 4. Save Model
    torch.save(model, os.path.join(save_path, "unet_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_weights.pth"))


    # 5. Plot Losses
    plot_losses(train_losses, val_losses, save_path)

    # 6. Save validation Dice scores to Excel
    #df_dice = pd.DataFrame(val_dice_all)
    #df_dice.to_excel(os.path.join(save_path, "validation_dice_scores.xlsx"), index=False)


    # 7. Testing
    test_dice = test(model, test_loader, device, save_path)
    test_dice_all.append(test_dice)

    # 8. Save Test Dice scores to Excel
    #df_test_dice = pd.DataFrame(test_dice_all)
    #df_test_dice.to_excel(os.path.join(save_path, "test_dice_scores.xlsx"), index=False)


    # 9. Visualize Predictions
    visualize_predictions(model, test_loader, device, save_path)