import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.stats import mannwhitneyu

# ===============================================================================================================
# Paths for each model's validation and test dice scores

# 2024 models
model_paths = OrderedDict({
    'Bing Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Claude 3.5 Sonnet': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Copilot': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Gemini 1.5 Pro': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4o': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o1 Preview': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Llama 3.1 405B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
    },
})

# 2025 Models
model_paths.update({
    'Claude 4 Sonnet': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'DeepSeek R1': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'DeepSeek V3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o4-mini-high': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Gemini 2.5 pro': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Grok 3 mini': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Grok 3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Llama 4 Maverick': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Mistral Medium 3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\BAGLS output\mask fix\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\BAGLS output\mask fix\test_dice_scores.xlsx"
    },
    'Qwen 3_235B': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
    },
})

# ===============================================================================================================

def load_dice_scores(model_paths):
    validation_scores = {}
    test_scores = {}

    for model, paths in model_paths.items():
        # Load & clean validation
        val = pd.read_excel(paths['validation'], header=None)
        fr = pd.to_numeric(val.iloc[0], errors='coerce')
        if fr.notna().all() and all(fr.astype(int)==np.arange(1,len(fr)+1)):
            val = val.iloc[1:]
        fc = pd.to_numeric(val.iloc[:,0], errors='coerce')
        if fc.notna().all() and all(fc.astype(int)==np.arange(1,len(fc)+1)):
            val = val.iloc[:,1:]
        nums = pd.to_numeric(val.values.flatten(), errors='coerce')
        nums = nums[~np.isnan(nums)]
        nums = nums[nums<1]
        validation_scores[model] = nums

        # Load & clean test
        tst = pd.read_excel(paths['test'], header=None)
        fr = pd.to_numeric(tst.iloc[0], errors='coerce')
        if fr.notna().all() and all(fr.astype(int)==np.arange(1,len(fr)+1)):
            tst = tst.iloc[1:]
        fc = pd.to_numeric(tst.iloc[:,0], errors='coerce')
        if fc.notna().all() and all(fc.astype(int)==np.arange(1,len(fc)+1)):
            tst = tst.iloc[:,1:]
        nums = pd.to_numeric(tst.values.flatten(), errors='coerce')
        nums = nums[~np.isnan(nums)]
        nums = nums[nums<1]
        test_scores[model] = nums

    return validation_scores, test_scores

validation_scores, test_scores = load_dice_scores(model_paths)

# ===============================================================================================================
# Compute medians for validation and test
medians_val  = {m: np.median(validation_scores[m]) for m in validation_scores}
medians_test = {m: np.median(test_scores[m])       for m in test_scores}

models_2024 = [
    'Bing Copilot', 'Claude 3.5 Sonnet', 'Copilot', 'Gemini 1.5 Pro',
    'GPT 4', 'GPT 4o', 'GPT o1 Preview', 'Llama 3.1 405B'
]
models_2025 = [
    'Claude 4 Sonnet', 'DeepSeek R1', 'DeepSeek V3', 'GPT o3',
    'GPT o4-mini-high', 'Gemini 2.5 pro', 'Grok 3 mini', 'Grok 3',
    'Llama 4 Maverick', 'Mistral Medium 3', 'Qwen 3_235B'
]

# 2) Compute medians
medians_val  = {m: np.median(validation_scores[m]) for m in validation_scores}
medians_test = {m: np.median(test_scores[m])       for m in test_scores}

# 3) Build plot orders
#    Requirement:
#    - nnU-Net always first (top)
#    - then 2024 models sorted by median amongst themselves
#    - then 2025 models sorted by median amongst themselves

plot_order_val_2024  = sorted(models_2024, key=lambda m: medians_val[m],  reverse=True)
plot_order_val_2025  = sorted(models_2025, key=lambda m: medians_val[m],  reverse=True)
plot_order_test_2024 = sorted(models_2024, key=lambda m: medians_test[m], reverse=True)
plot_order_test_2025 = sorted(models_2025, key=lambda m: medians_test[m], reverse=True)

plot_order_val  = ['nnU-Net'] + plot_order_val_2024  + plot_order_val_2025
plot_order_test = ['nnU-Net'] + plot_order_test_2024 + plot_order_test_2025

# 4) Colour map
color_map = {
    **{m: '#00BDD6' for m in models_2024},
    **{m: '#FF5E69' for m in models_2025},
    'nnU-Net': '#808080'
}

# ===============================================================================================================
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(11, 7))

# Validation subplot
val_list = [validation_scores[m] for m in plot_order_val]
bp = axes[0].boxplot(val_list, patch_artist=True, vert=False,
                     labels=plot_order_val, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order_val):
    patch.set_facecolor(color_map[model])
axes[0].set_title('BAGLS Dataset', fontsize=16)
axes[0].set_xlabel('Validation Dice Score', fontsize=14)
axes[0].invert_yaxis()   # first item at top
axes[0].set_xlim(0, 1)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[0].grid(True)

# Test subplot
test_list = [test_scores[m] for m in plot_order_test]
bp = axes[1].boxplot(test_list, patch_artist=True, vert=False,
                     labels=plot_order_test, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order_test):
    patch.set_facecolor(color_map[model])
axes[1].set_title('BAGLS Dataset', fontsize=16)
axes[1].set_xlabel('Dice Score', fontsize=14)
axes[1].invert_yaxis()
axes[1].set_xlim(0, 1)
axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[1].grid(True)

fig.suptitle('Model Dice Scores Comparison (BAGLS Dataset)', fontsize=22)
plt.tight_layout(rect=[0,0,1,0.96])
plt.subplots_adjust(wspace=0.7)

# Save
save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\new plots\combined\color coded unsorted\\"
plt.savefig(f"{save_path}all_model_dice_scores_combined_BAGLS.png", dpi=600)
