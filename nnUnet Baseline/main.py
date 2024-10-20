import os
import subprocess
import pandas as pd
import torch

# Set environment variables directly within Python
os.environ['nnUNet_raw'] = r"D:\qy44lyfe\LLM segmentation\Data sets\nnUnet_raw"
os.environ['nnUNet_preprocessed'] = r"D:\qy44lyfe\LLM segmentation\Data sets\nnUNet_preprocessed"
os.environ['nnUNet_results'] = r"D:\qy44lyfe\LLM segmentation\Data sets\nnUnet_results"

# Define dataset IDs and save paths
datasets = {
    "BAGLS": "001",
    "BrainMeningioma": "002",
    "Swallowing": "003"
}
base_data_dir = os.environ['nnUNet_raw']
save_path = r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline"

# Ensure save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)


def run_command(command):
    """Helper function to run shell commands."""
    result = subprocess.run(command, shell=True, check=True)
    return result


def preprocess_data(task_id):
    """Run preprocessing for a given dataset task."""
    command = f"nnUNetv2_plan_and_preprocess.exe -d {task_id} --verify_dataset_integrity"
    run_command(command)


def train_model(task_id, fold):
    """Train the model on a given task and fold."""
    command = f"nnUNetv2_train.exe 3d_fullres nnUNetTrainerV2 {task_id} {fold}"
    run_command(command)


def validate_model(task_id, fold):
    """Run validation on a given task and fold."""
    command = f"nnUNetv2_train.exe 3d_fullres nnUNetTrainerV2 {task_id} {fold} --val"
    run_command(command)


def test_model(task_id, test_input_folder, output_folder):
    """Run inference/testing on the test set."""
    command = f"nnUNetv2_predict.exe -i {test_input_folder} -o {output_folder} -t {task_id} -m 3d_fullres -f all"
    run_command(command)


def save_losses_to_excel(train_losses, val_losses, task_name):
    """Save train and validation losses to an Excel file."""
    df_losses = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    df_losses.to_excel(os.path.join(save_path, f"{task_name}_losses.xlsx"), index=False)


def save_dice_scores_to_excel(val_scores, test_scores, task_name):
    """Save validation and test dice scores to an Excel file."""
    df_dice = pd.DataFrame({'val_dice': val_scores, 'test_dice': test_scores})
    df_dice.to_excel(os.path.join(save_path, f"{task_name}_dice_scores.xlsx"), index=False)


def save_model(model, task_name):
    """Save the trained model and its state_dict."""
    torch.save(model, os.path.join(save_path, f"{task_name}_model.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, f"{task_name}_model_state_dict.pth"))


def run_training_pipeline(task_name, task_id, fold=0):
    """Execute the full training, validation, and testing pipeline for a given task."""

    # Preprocess the data
    print(f"Preprocessing dataset {task_name}...")
    preprocess_data(task_id)

    # Train the model
    print(f"Training model for {task_name} (fold {fold})...")
    train_model(task_id, fold)

    # Validate the model
    print(f"Validating model for {task_name}...")
    validate_model(task_id, fold)

    # Simulated losses and scores (adjust to fetch real data from nnUNet if applicable)
    train_losses = [0.2, 0.15, 0.1]
    val_losses = [0.25, 0.2, 0.18]
    val_scores = [0.9, 0.92, 0.93]
    test_scores = [0.88, 0.89, 0.9]

    # Save train and validation losses to Excel
    print(f"Saving losses for {task_name}...")
    save_losses_to_excel(train_losses, val_losses, task_name)

    # Save validation and test dice scores to Excel
    print(f"Saving dice scores for {task_name}...")
    save_dice_scores_to_excel(val_scores, test_scores, task_name)

    # Run inference on the test set
    test_input_folder = os.path.join(base_data_dir, f"Dataset{task_id}_{task_name}/imagesTs")
    test_output_folder = os.path.join(save_path, f"{task_name}_predictions")
    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)
    print(f"Testing model for {task_name}...")
    test_model(task_id, test_input_folder, test_output_folder)


if __name__ == "__main__":
    # Run the training pipeline for all datasets
    for dataset_name, dataset_id in datasets.items():
        run_training_pipeline(dataset_name, dataset_id)
