import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths for each model's validation and test dice scores

# BAGLS OUTPUT =========================================================================================================
model_paths = {
    'Bing Copilot': {
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
    'Llama 3.1 405B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
    }
}

# BOLUS OUTPUT =========================================================================================================
# model_paths = {
#     'Bing Copilot': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'Claude 3.5 Sonnet': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'Copilot': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'Gemini 1.5 Pro': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'GPT 4': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'GPT 4o': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'GPT o1 Preview': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'LLAMA 3.1 405B': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\Bolus output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\Bolus output\test_dice_scores.xlsx"
#     },
#     'nnUnet': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BOLUS.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BOLUS.xlsx"
#     }
# }


# BRAIN TUMOR OUTPUT ===================================================================================================
# model_paths = {
#     'Bing Copilot': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'Claude 3.5 Sonnet': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'Copilot': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'Gemini 1.5 Pro': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'GPT 4': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'GPT 4o': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'GPT o1 Preview': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'Llama 3.1 405B': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\test_dice_scores.xlsx"
#     },
#     'nnU-Net': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
#     }
# }
# ======================================================================================================================
#2025:
# model_paths = {
#     'Claude 4 Sonnet': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'DeepSeek R1': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'DeepSeek V3': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'GPT o3': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'GPT o4-mini-high': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'Gemini 2.5 pro': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'Grok 3 mini': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'Grok 3': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'Llama 4 Maverick': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'Mistral Medium 3': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\skin output\mask fix\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\skin output\mask fix\test_dice_scores.xlsx"
#     },
#     'Qwen 3_235B': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\skin output\validation_dice_scores.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\skin output\test_dice_scores.xlsx"
#     },
#     'nnU-Net': {
#         'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_SKIN.xlsx",
#         'test': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_SKIN.xlsx"
#     }
# }

# Function to load validation and test data from Excel files (handling inconsistent shapes)
def load_dice_scores(model_paths):
    validation_scores = {}
    test_scores = {}

    for model, paths in model_paths.items():
        # Load validation data
        #val_data = pd.read_excel(paths['validation'], sheet_name='Sheet1')
        #validation_scores[model] = val_data.mean(axis=1)  # Mean over batches (columns)

        # Load validation data, pick up any sheet
        val_data = pd.read_excel(paths['validation'], sheet_name=0)
        # grab ONLY the numeric columns, flatten them to 1D, drop NaNs
        val_nums = val_data.select_dtypes(include=[np.number]).values.flatten()
        val_nums = val_nums[~np.isnan(val_nums)]
        validation_scores[model] = val_nums

        # Load test data without headers
        #test_data = pd.read_excel(paths['test'], sheet_name='Sheet1', header=None)
        test_data = pd.read_excel(paths['test'], sheet_name=0)

        # Convert all data to numeric, coercing errors to NaN
        test_data = test_data.apply(pd.to_numeric, errors='coerce')

        # Stack the DataFrame to get all values in a single Series
        #test_series = test_data.stack().reset_index(drop=True)
        test_vals = test_data.values.flatten()
        test_vals = pd.to_numeric(test_vals, errors='coerce')
        test_vals = test_vals[~np.isnan(test_vals)]

        # # Remove NaN values
        # test_series = test_series.dropna()
        #
        # # If the first cell is "0", skip or ignore it
        # if not test_series.empty and test_series.iloc[0] == 0:
        #     test_series = test_series.iloc[1:]
        #
        # # Convert to numpy array
        # test_array = test_series.values
        #
        # # Store the test scores
        # test_scores[model] = test_array
        # Now test_vals is a clean 1D float array
        test_scores[model] = test_vals

    return validation_scores, test_scores

# Load the dice scores for all models
validation_scores, test_scores = load_dice_scores(model_paths)

# Calculate the average dice scores for both validation and test sets
avg_val_scores = {model: np.mean(scores) for model, scores in validation_scores.items()}
avg_test_scores = {model: np.mean(scores) for model, scores in test_scores.items()}

# Sort the models by average validation scores (from highest to lowest)
sorted_val_scores = dict(sorted(avg_val_scores.items(), key=lambda item: item[1], reverse=True))
sorted_test_scores = dict(sorted(avg_test_scores.items(), key=lambda item: item[1], reverse=True))

# Print the sorted validation dice scores
print("Average Validation Dice Scores from high to Low:")
for model, score in sorted_val_scores.items():
    print(f"{model}: {score:.4f}")

# Print the sorted test dice scores
print("\nAverage Test Dice Scores from high to low:")
for model, score in sorted_test_scores.items():
    print(f"{model}: {score:.4f}")

# Define the save path
save_path = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\new plots\\models 2024\\"

# Map colors to each model for consistency
# 2024:
color_map = {
    'Bing Copilot': '#66c2a5',
    'Claude 3.5 Sonnet': '#fc8d62',
    'Copilot': '#8da0cb',
    'Gemini 1.5 Pro': '#e78ac3',
    'GPT 4': '#a6d854',
    'GPT 4o': '#ffd92f',
    'GPT o1 Preview': '#e5c494',
    'Llama 3.1 405B': '#b3b3b3',
    'nnU-Net': '#1f78b4',
}

# # 2025:
# color_map = {
#     'Claude 4 Sonnet'       : '#66c2a5',  # teal (Set2[0])
#     'DeepSeek R1'           : '#fc8d62',  # orange (Set2[1])
#     'DeepSeek V3'           : '#8da0cb',  # light blue (Set2[2])
#     'GPT o3'                : '#e78ac3',  # pink (Set2[3])
#     'GPT o4-mini-high'      : '#a6d854',  # green (Set2[4])
#     'Gemini 2.5 pro'        : '#ffd92f',  # yellow (Set2[5])
#     'Grok 3 mini' : '#e5c494',  # tan (Set2[6])
#     'Grok 3'                : '#b3b3b3',  # grey (Set2[7])
#     'Llama 4 Maverick'      : '#e41a1c',  # red (added from ColorBrewer Set1)
#     'Mistral Medium 3'      : '#d95f02',  # dark orange (added from Set1)
#     'Qwen 3_235B'           : '#7570b3',  # purple (added from Set1)
#     'nnU-Net'               : '#1f78b4',  # your original blue
# }

# Create a figure with 1 row and 2 columns for the subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plotting Validation Dice Scores on the first subplot (axes[0])
val_data_list = [validation_scores[model] for model in model_paths.keys()]
boxplots = axes[0].boxplot(val_data_list, patch_artist=True, vert=False, labels=model_paths.keys(), showfliers=False)

# Apply colors based on the model-color map
for patch, model in zip(boxplots['boxes'], model_paths.keys()):
    patch.set_facecolor(color_map[model])

# Add titles and labels to the validation plot
axes[0].set_title('BAGLS Dataset', fontsize=14)
axes[0].set_xlabel('Validation Dice Scores', fontsize=12)
#axes[0].set_ylabel('Models', fontsize=12)
axes[0].grid(True)

# Plotting Test Dice Scores on the second subplot (axes[1])
test_data_list = [test_scores[model] for model in model_paths.keys()]
boxplots = axes[1].boxplot(test_data_list, patch_artist=True, vert=False, labels=model_paths.keys(), showfliers=False)

# Apply colors based on the model-color map
for patch, model in zip(boxplots['boxes'], model_paths.keys()):
    patch.set_facecolor(color_map[model])

# Add titles and labels to the test plot
axes[1].set_title('BAGLS Dataset', fontsize=14)
axes[1].set_xlabel('Test Dice Scores', fontsize=12)
axes[1].grid(True)

# Set the overall title for the figure
fig.suptitle('Model Dice Scores Comparison (BAGLS dataset)', fontsize=16)

# Adjust layout to ensure there's no overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.6)  # Increase space between the subplots

# Save the combined plot
plt.savefig(f"{save_path}all_model_dice_scores_baseline_BAGLS.png", dpi=600)

# Show the plot
#plt.show()