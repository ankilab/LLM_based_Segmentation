import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths for each model's validation and test dice scores
model_paths = {
    'Bing Microsoft Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Claude 3.5 Sonnet': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Gemini 1.5 Pro': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4o': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o1 Preview': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'LLAMA 3.1 405B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\test_dice_scores.xlsx"
    }
}


# Function to load validation and test data from Excel files
def load_dice_scores(model_paths):
    validation_scores = {}
    test_scores = {}

    for model, paths in model_paths.items():
        val_data = pd.read_excel(paths['validation'], sheet_name=None)
        test_data = pd.read_excel(paths['test'], sheet_name=None)

        # Assuming the data is in the first sheet for each model
        validation_scores[model] = val_data['Sheet1'].mean(axis=1)  # Mean over batches (columns)
        test_scores[model] = test_data['Sheet1'].values.flatten()  # Test dice scores

    return validation_scores, test_scores


# Load the dice scores for all models
validation_scores, test_scores = load_dice_scores(model_paths)

# Colors for each model
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB6C1', '#778899', '#FFFF66', '#9ACD32']

# Plotting Validation Dice Scores
plt.figure(figsize=(8, 6))  # Smaller size for a paper

# Prepare the data for boxplots
val_data_list = [validation_scores[model] for model in model_paths.keys()]

# Create boxplots with custom colors and remove outliers
boxplots = plt.boxplot(val_data_list, patch_artist=True, labels=model_paths.keys(), showfliers=False)

# Apply different colors to each box
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Add titles, labels, and grid for better readability
plt.title('Validation Dice Scores Comparison (Mean over Epochs)', fontsize=14)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Validation Dice Scores', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Adding a legend manually with the same colors for each model (placed outside)
plt.legend([boxplots["boxes"][i] for i in range(len(model_paths))], model_paths.keys(), loc='center left',
           bbox_to_anchor=(1, 0.5), fontsize=9)

# Display the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
plt.show()

# Plotting Test Dice Scores
plt.figure(figsize=(8, 6))  # Smaller size for a paper

# Prepare the data for boxplots
test_data_list = [test_scores[model] for model in model_paths.keys()]

# Create boxplots with custom colors and remove outliers
boxplots = plt.boxplot(test_data_list, patch_artist=True, labels=model_paths.keys(), showfliers=False)

# Apply different colors to each box
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Add titles, labels, and grid for better readability
plt.title('Test Dice Scores Comparison', fontsize=14)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Test Dice Scores', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Adding a legend manually with the same colors for each model (placed outside)
plt.legend([boxplots["boxes"][i] for i in range(len(model_paths))], model_paths.keys(), loc='center left',
           bbox_to_anchor=(1, 0.5), fontsize=9)

# Display the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
plt.show()
