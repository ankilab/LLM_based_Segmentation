import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset
import torch.optim as optim
from model import UNet
from train import train_epoch, validate_epoch, test_model, save_losses, plot_losses, visualize_results, DiceBCELoss, \
    dice_score
import time
from torchinfo import summary
from torchvision import transforms

# Configuration
DATA_PATH = 'D:\qy44lyfe\LLM segmentation\Data sets\Swallowing'  # Update this path
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
MASK_DIR = os.path.join(DATA_PATH, 'masks')
SAVE_PATH = 'D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\DeepSeek R1\\out of the box\\Bolus output'  # Directory to save outputs
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42
IMG_SIZE = 256

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load dataset
dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transform=transform)

# Split dataset
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=SEED)

print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")

# Create subsets
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

# DataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Print model summary
#summary(model, input_size=(1, 1, IMG_SIZE, IMG_SIZE)) # correction next line
summary(model, input_size=(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE))

# Training loop
train_losses, val_losses = [], []
val_dice_scores = []

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, dice_scores = validate_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_dice_scores.append(dice_scores)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {np.mean(dice_scores):.4f}")

# Calculate total training time
total_time = time.time() - start_time
print(f"\nTotal training time: {total_time // 60:.0f}m {total_time % 60:.0f}s")

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
torch.save(model, os.path.join(SAVE_PATH, 'full_model.pth'))

# Save losses and plot
save_losses(train_losses, val_losses, SAVE_PATH)
plot_losses(train_losses, val_losses, SAVE_PATH)

# Save validation dice scores
val_dice_df = pd.DataFrame(val_dice_scores)
val_dice_df.to_excel(os.path.join(SAVE_PATH, 'validation_dice_scores.xlsx'), index=False)

# Test the model
test_dice = test_model(model, test_loader, device)
test_dice_df = pd.DataFrame([test_dice])
test_dice_df.to_excel(os.path.join(SAVE_PATH, 'test_dice_scores.xlsx'), index=False)
print(f"Test Dice Score: {np.mean(test_dice):.4f}")

# Visualize results
visualize_results(model, test_loader, device, SAVE_PATH)