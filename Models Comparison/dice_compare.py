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

# Function to load validation and test data from Excel files (handling inconsistent shapes)
def load_dice_scores(model_paths):
    validation_scores = {}
    test_scores = {}

    for model, paths in model_paths.items():
        # Load validation data
        val_data = pd.read_excel(paths['validation'], sheet_name='Sheet1')
        validation_scores[model] = val_data.mean(axis=1)  # Mean over batches (columns)

        # Load test data without headers
        test_data = pd.read_excel(paths['test'], sheet_name='Sheet1', header=None)
        print(f"{model} test data: {test_data}")

        # Convert all data to numeric, coercing errors to NaN
        test_data = test_data.apply(pd.to_numeric, errors='coerce')

        # Stack the DataFrame to get all values in a single Series
        test_series = test_data.stack().reset_index(drop=True)

        # Remove NaN values
        test_series = test_series.dropna()

        # If the first cell is "0", skip or ignore it
        if not test_series.empty and test_series.iloc[0] == 0:
            test_series = test_series.iloc[1:]

        # Convert to numpy array
        test_array = test_series.values

        # Store the test scores
        test_scores[model] = test_array

    return validation_scores, test_scores

# Load the dice scores for all models
validation_scores, test_scores = load_dice_scores(model_paths)

# Define the save path
save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\"

# Colors for each model
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB6C1', '#778899', '#FFFF66', '#9ACD32']

# Plotting Validation Dice Scores (Horizontal Box Plot)
plt.figure(figsize=(8, 4))  # Smaller size for a paper

# Prepare the data for boxplots
val_data_list = [validation_scores[model] for model in model_paths.keys()]

# Create horizontal boxplots with custom colors and remove outliers
boxplots = plt.boxplot(val_data_list, patch_artist=True, vert=False, labels=model_paths.keys(), showfliers=False)

# Apply different colors to each box
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Add titles, labels, and grid for better readability
plt.title('Validation Dice Scores Comparison (Mean over Epochs)', fontsize=14)
plt.xlabel('Validation Dice Scores', fontsize=12)
plt.ylabel('Models', fontsize=12)
plt.grid(True)

# Display the plot
plt.tight_layout()
# Save the test plot
plt.savefig(f"{save_path}model_comparison_validation dice_BAGLS.png", dpi=600)
#plt.show()


# Plotting Test Dice Scores (Horizontal Box Plot)
plt.figure(figsize=(8, 4))  # Smaller size for a paper

# Prepare the data for boxplots
test_data_list = [test_scores[model] for model in model_paths.keys()]

# Create horizontal boxplots with custom colors and remove outliers
boxplots = plt.boxplot(test_data_list, patch_artist=True, vert=False, labels=model_paths.keys(), showfliers=False)

# Apply different colors to each box
for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Add titles, labels, and grid for better readability
plt.title('Test Dice Scores Comparison', fontsize=14)
plt.xlabel('Test Dice Scores', fontsize=12)
plt.ylabel('Models', fontsize=12)
plt.grid(True)

# Display the plot
plt.tight_layout()
# Save the test plot
plt.savefig(f"{save_path}model_comparison_test Dice_BAGLS.png", dpi=600)
#plt.show()
