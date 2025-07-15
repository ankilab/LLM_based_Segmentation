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
train_loss_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2025\\all_train_losses_myoma.xlsx"
val_loss_path   = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\models 2025\\all_validation_losses_myoma.xlsx"
# =====================================================================================================================

save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\new plots\\models 2025\\test\\"

# Read both Excel files
train_loss_data = pd.read_excel(train_loss_path, sheet_name='Tabelle1')
val_loss_data   = pd.read_excel(val_loss_path,   sheet_name='Tabelle1')

# Function to prepare the data for plotting (extract model names and their corresponding losses)
def prepare_loss_data(loss_data):
    models = loss_data.iloc[:, 0]      # Model names are in the first column
    losses = loss_data.iloc[:, 1:]     # Loss values are in the remaining columns
    return models, losses

# Prepare training and validation losses
train_models, train_losses = prepare_loss_data(train_loss_data)
val_models,   val_losses   = prepare_loss_data(val_loss_data)

# Map colors to each model for consistency
color_map = {
    'Claude 4 Sonnet'       : '#66c2a5',
    'DeepSeek R1'           : '#fc8d62',
    'DeepSeek V3'           : '#8da0cb',
    'GPT o3'                : '#e78ac3',
    'GPT o4-mini-high'      : '#a6d854',
    'Gemini 2.5 pro'        : '#ffd92f',
    'Grok 3 mini'           : '#e5c494',
    'Grok 3'                : '#b3b3b3',
    'Llama 4 Maverick'      : '#e41a1c',
    'Mistral Medium 3'      : '#d95f02',
    'Qwen 3_235B'           : '#7570b3',
    'nnU-Net'               : '#1f78b4',
}

### Normal Scale Plot ### ==============================================================================
# Create a Figure + GridSpec: 1 row, 3 cols (2 plots + 1 legend panel)
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.25], wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
legend_ax = fig.add_subplot(gs[0, 2])
legend_ax.axis('off')  # hide axes for legend panel

# Plotting Training Losses on ax0 (no legend)
for i, model in enumerate(train_models):
    epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)
    ax0.plot(epochs, train_losses.iloc[i].dropna(), color=color_map[model])
ax0.set_title('Uterine Myoma Dataset', fontsize=16)
ax0.set_xlabel('Epoch', fontsize=14)
ax0.set_ylabel('Training Loss', fontsize=14)
ax0.grid(True)

# Plotting Validation Losses on ax1
for i, model in enumerate(val_models):
    epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
    ax1.plot(epochs, val_losses.iloc[i].dropna(), label=model, color=color_map[model])
ax1.set_title('Uterine Myoma Dataset', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Validation Loss', fontsize=14)
ax1.grid(True)

# Build and draw a single legend in legend_ax with a light, semi-transparent frame
handles, labels = ax1.get_legend_handles_labels()
lg = legend_ax.legend(
    handles, labels,
    loc='center',
    frameon=True
)
frame = lg.get_frame()
frame.set_alpha(0.5)                # semi-transparent
frame.set_facecolor('white')        # white background
frame.set_edgecolor('gray')         # light gray border

fig.suptitle('Loss Comparison (Myoma dataset)', fontsize=16)
plt.savefig(f"{save_path}2025_all_model_losses_baseline_total_MYOMA_legend.png", dpi=600)
# plt.show()


### Logarithmic Scale Plot ### =====================================================================
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.25], wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
legend_ax = fig.add_subplot(gs[0, 2])
legend_ax.axis('off')

yticks_custom = [0.001, 0.01, 0.1, 1]

# Training Losses (log scale) on ax0 (no legend)
for i, model in enumerate(train_models):
    epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)
    ax0.plot(epochs, train_losses.iloc[i].dropna(), color=color_map[model])
ax0.set_title('Uterine Myoma Dataset', fontsize=16)
ax0.set_xlabel('Epoch', fontsize=14)
ax0.set_ylabel('Training Loss', fontsize=14)
ax0.set_yscale('log')
ax0.set_yticks(yticks_custom)
ax0.grid(True)

# Validation Losses (log scale) on ax1
for i, model in enumerate(val_models):
    epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
    ax1.plot(epochs, val_losses.iloc[i].dropna(), label=model, color=color_map[model])
ax1.set_title('Uterine Myoma Dataset', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Validation Loss', fontsize=14)
ax1.set_yscale('log')
ax1.set_yticks(yticks_custom)
ax1.grid(True)

# Single legend in legend_ax with semi-transparent frame
handles, labels = ax1.get_legend_handles_labels()
lg = legend_ax.legend(
    handles, labels,
    loc='center',
    frameon=True
)
frame = lg.get_frame()
frame.set_alpha(0.5)
frame.set_facecolor('white')
frame.set_edgecolor('gray')

fig.suptitle('Loss Comparison (Uterine Myoma Dataset)', fontsize=16)
plt.savefig(f"{save_path}2025_all_model_losses_logarithmic_baseline_total_Myoma_legend.png", dpi=600)
# plt.show()
