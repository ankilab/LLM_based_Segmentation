import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths for each model's training and validation loss files
model_paths = {
    'Bing Microsoft Copilot': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\val_losses.xlsx"
    },
    'Claude 3.5 Sonnet': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\val_losses.xlsx"
    },
    'Copilot': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\val_losses.xlsx"
    },
    'Gemini 1.5 Pro': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\val_losses.xlsx"
    },
    'GPT 4': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\val_losses.xlsx"
    },
    'GPT 4o': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\val_losses.xlsx"
    },
    'GPT o1 Preview': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\val_losses.xlsx"
    },
    'LLAMA 3.1 405B': {
        'training': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\train_losses.xlsx",
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\val_losses.xlsx"
    }
}


# Function to load loss data from Excel files
def load_loss_data(filepath):
    # Read the Excel file without headers
    try:
        data = pd.read_excel(filepath, header=None)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return np.array([])

    # Drop entirely empty rows and columns
    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')

    # Replace zeros with NaN to ignore them
    data = data.replace(0, np.nan)

    # Convert all entries to numeric, coercing errors to NaN
    data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

    # Remove rows and columns that are now entirely NaN
    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')

    # Now process based on the shape of the data
    if data.shape[1] == 2:
        # Data has two columns, possibly epoch numbers and loss values
        # Check if the first column contains integers starting from 1
        first_column = data.iloc[:, 0].values
        if np.all(first_column == np.arange(1, len(first_column) + 1)):
            # The loss values are in the second column
            loss_values = data.iloc[:, 1].values
            return loss_values
        else:
            # Not epoch numbers, flatten the data
            loss_values = data.values.flatten()
            return loss_values
    elif data.shape[0] == 2:
        # Data has two rows, possibly epoch numbers and loss values
        first_row = data.iloc[0, :].values
        if np.all(first_row == np.arange(1, len(first_row) + 1)):
            # The loss values are in the second row
            loss_values = data.iloc[1, :].values
            return loss_values
        else:
            # Not epoch numbers, flatten the data
            loss_values = data.values.flatten()
            return loss_values
    else:
        # Data is in other shape, flatten to 1D array
        loss_values = data.values.flatten()
        return loss_values


# Initialize dictionaries to store loss values per model
training_losses = {}
validation_losses = {}

# Load the loss data for each model
for model_name, paths in model_paths.items():
    # Load training loss data
    training_loss = load_loss_data(paths['training'])
    training_losses[model_name] = training_loss

    # Load validation loss data
    validation_loss = load_loss_data(paths['validation'])
    validation_losses[model_name] = validation_loss

    # Print statements to verify the data
    print(f"{model_name} training loss: {training_loss}")
    print(f"{model_name} validation loss: {validation_loss}")

# Colors for each model
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

# Plot the training losses
plt.figure(figsize=(10, 6))

for i, (model_name, loss_values) in enumerate(training_losses.items()):
    if len(loss_values) == 0:
        continue  # Skip if no data
    epochs = np.arange(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label=model_name, color=colors[i % len(colors)])

plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the validation losses
plt.figure(figsize=(10, 6))

for i, (model_name, loss_values) in enumerate(validation_losses.items()):
    if len(loss_values) == 0:
        continue  # Skip if no data
    epochs = np.arange(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label=model_name, color=colors[i % len(colors)])

plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()