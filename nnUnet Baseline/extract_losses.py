import re
import pandas as pd
import argparse

# Define the path to log file
#log_file_path = r"Z:\BVM_LM_datasets\nnunet_train\nnUNet_results\Dataset111_BAGLS\nnUNetTrainer__nnUNetPlans__2d\fold_6\training_log_2024_10_22_13_50_38.txt"
#log_file_path = r"Z:\BVM_LM_datasets\nnunet_train\nnUNet_results\Dataset113_Swallowing\nnUNetTrainer__nnUNetPlans__2d\fold_6\training_log_2024_10_23_09_35_06.txt"
#log_file_path = r"Z:\BVM_LM_datasets\nnunet_train\nnUNet_results\Dataset112_BrainMeningioma\nnUNetTrainer__nnUNetPlans__2d\fold_6\training_log_2024_10_23_09_39_00.txt"
#log_file_path = r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\Baseline 2025\nnunet_train\nnUNet_results\Dataset116_MYOMA\nnUNetTrainer_nnUNetPlans_2d\fold_0\training_log_2025_7_8_13_45_18.txt"

# Define the path to save the output CSV file
#save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\parsed_metrics_Myoma.xlsx"

def parse_log(log_file_path: str, save_path: str):
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

    # Create DataFrame and save to Excel
    metrics_df = pd.DataFrame({
        "Epoch": epochs,
        "Training Loss": train_losses,
        "Validation Loss": val_losses,
        "Dice Score": dice_scores
    })
    metrics_df.to_excel(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Parse nnU-Net training log and extract epoch, losses, and Dice metrics."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to the nnU-Net training log file"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path where the parsed metrics Excel file will be saved"
    )
    args = parser.parse_args()

    parse_log(args.log_file, args.save_path)

if __name__ == "__main__":
    main()
