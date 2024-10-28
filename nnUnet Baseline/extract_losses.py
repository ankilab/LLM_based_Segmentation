import re
import pandas as pd

# Define the path to your log file
log_file_path = r"Z:\BVM_LM_datasets\nnunet_train\nnUNet_results\Dataset111_BAGLS\nnUNetTrainer__nnUNetPlans__2d\fold_6\training_log_2024_10_22_13_50_38.txt"

# Define the path to save the output CSV file
save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\parsed_metrics.xlsx"

# Initialize lists to store metrics
epochs = []
train_losses = []
val_losses = []
dice_scores = []

# Open and read the log file
with open(log_file_path, "r") as file:
    for line in file:
        # Match lines containing epoch number
        epoch_match = re.search(r"Epoch (\d+)", line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        # Match lines containing training loss
        train_loss_match = re.search(r"train_loss ([\-\d.]+)", line)
        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))

        # Match lines containing validation loss
        val_loss_match = re.search(r"val_loss ([\-\d.]+)", line)
        if val_loss_match:
            val_losses.append(float(val_loss_match.group(1)))

        # Match lines containing Dice score
        dice_score_match = re.search(r"Pseudo dice \[np\.float32\(([\d.]+)\)\]", line)
        if dice_score_match:
            dice_scores.append(float(dice_score_match.group(1)))

# Create DataFrame and save to CSV
metrics_df = pd.DataFrame({
    "Epoch": epochs,
    "Training Loss": train_losses,
    "Validation Loss": val_losses,
    "Dice Score": dice_scores
})
metrics_df.to_excel(save_path, index=False)
print(f"Metrics saved to {save_path}")