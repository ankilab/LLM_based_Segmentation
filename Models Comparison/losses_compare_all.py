import pandas as pd
import matplotlib.pyplot as plt

# Load training and validation losses
# BAGLS ===============================================================================================================
#train_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2024\\all_train_losses_Retina.xlsx"
#val_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2024\\all_validation_losses_Retina.xlsx"
# BOLUS ===============================================================================================================
# train_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\all_train_losses_BOLUS.xlsx"
# val_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\all_validation_losses_BOLUS.xlsx"
# BRAIN ===============================================================================================================
train_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2025\\all_train_losses_SKIN.xlsx"
val_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2025\\all_validation_losses_SKIN.xlsx"
# =====================================================================================================================

save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\new plots\\models 2025\\"

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

# Map colors to each model for consistency

##2024 models:
# color_map = {
#     'Bing Microsoft Copilot': '#66c2a5',
#     'Claude 3.5 Sonnet': '#fc8d62',
#     'Copilot': '#8da0cb',
#     'Gemini 1.5 Pro': '#e78ac3',
#     'GPT4': '#a6d854',
#     'GPT 4o': '#ffd92f',
#     'GPT o1 Preview': '#e5c494',
#     'Llama 3.1 405B': '#b3b3b3',
#     'nnU-Net': '#1f78b4'
# }

# 2025 models:

color_map = {
    'Claude 4 Sonnet'       : '#66c2a5',  # teal (Set2[0])
    'DeepSeek R1'           : '#fc8d62',  # orange (Set2[1])
    'DeepSeek V3'           : '#8da0cb',  # light blue (Set2[2])
    'GPT o3'                : '#e78ac3',  # pink (Set2[3])
    'GPT o4-mini-high'      : '#a6d854',  # green (Set2[4])
    'Gemini 2.5 pro'        : '#ffd92f',  # yellow (Set2[5])
    'Grok 3 mini' : '#e5c494',  # tan (Set2[6])
    'Grok 3'                : '#b3b3b3',  # grey (Set2[7])
    'Llama 4 Maverick'      : '#e41a1c',  # red (added from ColorBrewer Set1)
    'Mistral Medium 3'      : '#d95f02',  # dark orange (added from Set1)
    'Qwen 3_235B'           : '#7570b3',  # purple (added from Set1)
    'nnU-Net'               : '#1f78b4',  # your original blue
}

### Normal Scale Plot ### ==============================================================================
# Create a figure with 1 row and 2 columns for the subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plotting Training Losses on the first subplot (axes[0])
for i, model in enumerate(train_models):
    epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[0].plot(epochs, train_losses.iloc[i].dropna(), label=model, color=color_map[model])
axes[0].set_title('Training Loss per Epoch', fontsize=14)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Training Loss', fontsize=12)
axes[0].legend()
axes[0].grid(True)

# Plotting Validation Losses on the second subplot (axes[1])
for i, model in enumerate(val_models):
    epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[1].plot(epochs, val_losses.iloc[i].dropna(), label=model, color=color_map[model])
axes[1].set_title('Validation Loss per Epoch', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Loss', fontsize=12)
axes[1].legend()
axes[1].grid(True)

# Set the overall title for the figure
fig.suptitle('Loss Comparison (Skin Lesion dataset)', fontsize=16)

# Adjust layout to ensure there's no overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
#plt.show()
# Save the combined plot
plt.savefig(f"{save_path}2025_all_model_losses_baseline_total_SKIN.png", dpi=600)

### Logarithmic Scale Plot ### ===========================================================================
# Create a figure with 1 row and 2 columns for the subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Define more detailed custom y-ticks for the logarithmic scale
yticks_custom = [0.001, 0.01, 0.1, 1]

# Plotting Training Losses on the first subplot (axes[0])
for i, model in enumerate(train_models):
    epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[0].plot(epochs, train_losses.iloc[i].dropna(), label=model, color=color_map[model])
axes[0].set_title('Training Loss per Epoch', fontsize=14)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Training Loss (log scale)', fontsize=12)
axes[0].set_yscale('log')  # Set y-axis to logarithmic scale
axes[0].set_yticks(yticks_custom)  # Apply more y-ticks to fill in the gaps
axes[0].legend()
axes[0].grid(True)

# Plotting Validation Losses on the second subplot (axes[1])
for i, model in enumerate(val_models):
    epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)  # Handle varying epoch lengths
    axes[1].plot(epochs, val_losses.iloc[i].dropna(), label=model, color=color_map[model])
axes[1].set_title('Validation Loss per Epoch', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Loss (log scale)', fontsize=12)
axes[1].set_yscale('log')  # Set y-axis to logarithmic scale
axes[1].set_yticks(yticks_custom)  # Apply more y-ticks to fill in the gaps
axes[1].legend()
axes[1].grid(True)

# Set the overall title for the figure
fig.suptitle('Loss Comparison (Skin Lesion Dataset)', fontsize=16)

# Adjust layout to ensure there's no overlap and more space between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{save_path}2025_all_model_losses_logarithmic_baseline_total_SKIN.png", dpi=600)
#plt.show()
