import pandas as pd
import matplotlib.pyplot as plt

# Load training and validation losses
train_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\all_train_losses_BAGLS.xlsx"
val_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\all_validation_losses_BAGLS.xlsx"

# Read both Excel files
train_loss_data = pd.read_excel(train_loss_path, sheet_name='Tabelle1')
val_loss_data = pd.read_excel(val_loss_path, sheet_name='Tabelle1')

# Function to prepare the data for plotting (extract model names and their corresponding losses)
def prepare_loss_data(loss_data):
    models = loss_data.iloc[:, 0]  # Model names are in the first column
    losses = loss_data.iloc[:, 1:]  # Loss values are in the remaining columns
    return models, losses

# Prepare training and validation losses
train_models, train_losses = prepare_loss_data(train_loss_data)
val_models, val_losses = prepare_loss_data(val_loss_data)

# Colors for each model (can be adjusted if more models are added)
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB6C1', '#778899', '#FFFF66', '#9ACD32']

# Create a figure with 1 row and 2 columns for the subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting Training Losses on the first subplot (axes[0])
for i, model in enumerate(train_models):
    epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[0].plot(epochs, train_losses.iloc[i].dropna(), label=model, color=colors[i % len(colors)])
axes[0].set_title('Training Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].legend()
axes[0].grid(True)

# Plotting Validation Losses on the second subplot (axes[1])
for i, model in enumerate(val_models):
    epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[1].plot(epochs, val_losses.iloc[i].dropna(), label=model, color=colors[i % len(colors)])
axes[1].set_title('Validation Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].legend()
axes[1].grid(True)

# Set the overall title for the figure
fig.suptitle('Loss Comparison', fontsize=16)

# Adjust layout to ensure there's no overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.show()
