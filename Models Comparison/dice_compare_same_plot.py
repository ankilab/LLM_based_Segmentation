import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# ===============================================================================================================
# Paths for each model's validation and test dice scores

# 2024 models
model_paths = OrderedDict({
    'Bing Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'Claude 3.5 Sonnet': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'Gemini 1.5 Pro': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'GPT 4': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\Retina output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\Retina output\test_dice_scores.xlsx"
    },
    'GPT 4o': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'GPT o1 Preview': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'Llama 3.1 405B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\utrine myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\utrine myoma output\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_Myoma.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_Myoma.xlsx"
    },
})

# 2025 Models
model_paths.update({
    'Claude 4 Sonnet': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'DeepSeek R1': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'DeepSeek V3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'GPT o3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'GPT o4-mini-high': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'Gemini 2.5 pro': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'Grok 3 mini': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'Grok 3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'Llama 4 Maverick': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'Mistral Medium 3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\Myoma output\mask fix\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\Myoma output\mask fix\test_dice_scores.xlsx"
    },
    'Qwen 3_235B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\Myoma output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\Myoma output\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_Myoma.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_Myoma.xlsx"
    },
})

# ===============================================================================================================

def load_dice_scores(model_paths):
    validation_scores = {}
    test_scores = {}

    for model, paths in model_paths.items():
        # --- Validation ---
        val = pd.read_excel(paths['validation'], header=None)
        fr = pd.to_numeric(val.iloc[0], errors='coerce')
        if fr.notna().all() and all(fr.astype(int) == np.arange(1, len(fr)+1)):
            val = val.iloc[1:]
        fc = pd.to_numeric(val.iloc[:, 0], errors='coerce')
        if fc.notna().all() and all(fc.astype(int) == np.arange(1, len(fc)+1)):
            val = val.iloc[:, 1:]
        flat = val.values.flatten()
        nums = pd.to_numeric(flat, errors='coerce')
        nums = nums[~np.isnan(nums)]
        nums = nums[nums < 1]
        validation_scores[model] = nums

        # --- Test ---
        tst = pd.read_excel(paths['test'], header=None)
        fr = pd.to_numeric(tst.iloc[0], errors='coerce')
        if fr.notna().all() and all(fr.astype(int) == np.arange(1, len(fr)+1)):
            tst = tst.iloc[1:]
        fc = pd.to_numeric(tst.iloc[:, 0], errors='coerce')
        if fc.notna().all() and all(fc.astype(int) == np.arange(1, len(fc)+1)):
            tst = tst.iloc[:, 1:]
        flat = tst.values.flatten()
        nums = pd.to_numeric(flat, errors='coerce')
        nums = nums[~np.isnan(nums)]
        nums = nums[nums < 1]
        test_scores[model] = nums

    return validation_scores, test_scores

validation_scores, test_scores = load_dice_scores(model_paths)

# ===============================================================================================================
# Color map for all models
color_map = {
    # 2024
    'Bing Copilot':   '#66c2a5',
    'Claude 3.5 Sonnet': '#fc8d62',
    'Copilot':        '#8da0cb',
    'Gemini 1.5 Pro': '#e78ac3',
    'GPT 4':          '#a6d854',
    'GPT 4o':         '#ffd92f',
    'GPT o1 Preview': '#e5c494',
    'Llama 3.1 405B': '#b3b3b3',
    'nnU-Net':        '#1f78b4',
    # 2025
    'Claude 4 Sonnet':  '#66c2a5',
    'DeepSeek R1':      '#fc8d62',
    'DeepSeek V3':      '#8da0cb',
    'GPT o3':           '#e78ac3',
    'GPT o4-mini-high': '#a6d854',
    'Gemini 2.5 pro':   '#ffd92f',
    'Grok 3 mini':      '#e5c494',
    'Grok 3':           '#b3b3b3',
    'Llama 4 Maverick': '#e41a1c',
    'Mistral Medium 3': '#d95f02',
    'Qwen 3_235B':      '#7570b3',
}

# Build a reversed-2024 + reversed-2025 key order in two lines:
keys = list(model_paths.keys())
n2024 = 9
plot_order = keys[:n2024][::-1] + keys[n2024:][::-1]

# ===============================================================================================================
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(11, 7))

# Validation subplot
val_list = [validation_scores[m] for m in plot_order]
bp = axes[0].boxplot(val_list, patch_artist=True, vert=False,
                     labels=plot_order, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order):
    patch.set_facecolor(color_map[model])
axes[0].set_title('Uterine Myoma Dataset\nValidation Dice Scores', fontsize=16)
axes[0].set_xlabel('Dice Score', fontsize=14)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True)
axes[0].set_xlim(0, 1)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[0].invert_yaxis()

# Test subplot
test_list = [test_scores[m] for m in plot_order]
bp = axes[1].boxplot(test_list, patch_artist=True, vert=False,
                     labels=plot_order, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order):
    patch.set_facecolor(color_map[model])
axes[1].set_title('Uterine Myoma Dataset\nTest Dice Scores', fontsize=16)
axes[1].set_xlabel('Dice Score', fontsize=14)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].grid(True)
axes[1].set_xlim(0, 1)
axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[1].invert_yaxis()

fig.suptitle('Model Dice Scores Comparison (Uterine Myoma Dataset)', fontsize=22)
plt.tight_layout(rect=[0,0,1,0.96])
plt.subplots_adjust(wspace=0.7)

# Save
save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\new plots\combined\\"
plt.savefig(f"{save_path}all_model_dice_scores_combined_Myoma.png", dpi=600)
# plt.show()
