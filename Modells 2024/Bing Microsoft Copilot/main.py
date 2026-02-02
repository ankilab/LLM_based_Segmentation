import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
from model import UNet
from torchvision import transforms
from train import train_model, visualize_predictions, test
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

if __name__ == "__main__":
    #image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset"
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_images\converted"
    #image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"

    #mask_folder = "D:\\qy44lyfe\\LLM segmentation\\Data sets\\BAGLS\\subset"
    mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_masks\converted"
    #mask_folder = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"

    #save_path = "D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output"
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\Retina output"
    #save_path = "D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\Brain output"

    os.makedirs(save_path, exist_ok=True)

    # dataset = SegmentationDataset(image_dir, transform=transforms.ToTensor())
    # train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    # val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    #
    # train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet().to(device)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #
    # num_epochs = 25
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)
    #
    # print(f"Training completed. Model and results saved to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Define the target size for resizing
    target_size = (256, 256)

    dataset = SegmentationDataset(image_dir, mask_folder, transform=transforms.ToTensor(), target_size=target_size)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Print sizes
    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Print model summary using torchinfo
    summary(model, input_size=(16, 1, 256, 256))  # 1 channel for grayscale

    num_epochs = 25

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    print(f"Training completed. Model and results saved to {save_path}")

    test(model, test_loader, device, save_path)
    # Visualize predictions
    visualize_predictions(model, test_loader, device, save_path)
