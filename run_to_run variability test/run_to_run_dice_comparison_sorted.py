import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.stats import mannwhitneyu
import matplotlib.patches as mpatches

# ===============================================================================================================
# Paths for each model's validation and test dice scores

# 2024 models
model_paths = OrderedDict({
    'GPT 4o-1': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-1\test_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-1\test_dice_scores.xlsx"
    },
    'GPT 4o-2': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-2\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-2\test_dice_scores.xlsx"
    },
    'GPT 4o-3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-3\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-3\test_dice_scores.xlsx"
    },
    'GPT 4o-4': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-4\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-4\test_dice_scores.xlsx"
    },
    'GPT 4o-5': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-5\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-5\test_dice_scores.xlsx"
    },
    'GPT 4o-6': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-6\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-6\test_dice_scores.xlsx"
    },
    'GPT 4o-7': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-7\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-7\test_dice_scores.xlsx"
    },
    'GPT 4o-8': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-8\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-8\test_dice_scores.xlsx"
    },
    'GPT 4o-9': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-9\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-9\test_dice_scores.xlsx"
    },
    'GPT 4o-10': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-10\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT 4o-10\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BRAIN.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BRAIN.xlsx"
    },
})

# 2025 Models
model_paths.update({
    'GPT o4 mini-1': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-1\test_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-1\test_dice_scores.xlsx"
    },
    'GPT o4 mini-2': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-2\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-2\test_dice_scores.xlsx"
    },
    'GPT o4 mini-3': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-3\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-3\test_dice_scores.xlsx"
    },
    'GPT o4 mini-4': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-4\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-4\test_dice_scores.xlsx"
    },
    'GPT o4 mini-5': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-5\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-5\test_dice_scores.xlsx"
    },
    'GPT o4 mini-6': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-6\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-6\test_dice_scores.xlsx"
    },
    'GPT o4 mini-7': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-7\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-7\test_dice_scores.xlsx"
    },
    'GPT o4 mini-8': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-8\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-8\test_dice_scores.xlsx"
    },
    'GPT o4 mini-9': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-9\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-9\test_dice_scores.xlsx"
    },
    'GPT o4 mini-10': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-10\test_dice_scores.xlsx",
        'test': r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-10\test_dice_scores.xlsx"
    },
    'nnU-Net': {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BRAIN.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BRAIN.xlsx"
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
    'GPT 4o-1', 'GPT 4o-2', 'GPT 4o-3', 'GPT 4o-4',
    'GPT 4o-5', 'GPT 4o-6', 'GPT 4o-7', 'GPT 4o-8'
    ,'GPT 4o-9','GPT 4o-10'
]
models_2025 = [
    'GPT o4 mini-1', 'GPT o4 mini-2', 'GPT o4 mini-3', 'GPT o4 mini-4',
    'GPT o4 mini-5', 'GPT o4 mini-6', 'GPT o4 mini-7', 'GPT o4 mini-8'
    ,'GPT o4 mini-9','GPT o4 mini-10'
]

# 2) Compute medians
medians_val  = {m: np.median(validation_scores[m]) for m in validation_scores}
medians_test = {m: np.median(test_scores[m])       for m in test_scores}

# 3) Build plot orders
other_models = [m for m in model_paths if m != 'nnU-Net']
plot_order_val  = ['nnU-Net'] + sorted(other_models, key=lambda m: medians_val[m],  reverse=True)
plot_order_test = ['nnU-Net'] + sorted(other_models, key=lambda m: medians_test[m], reverse=True)

# 4) Colour map
color_map = {
    **{m: '#00BDD6' for m in models_2024},
    **{m: '#FF5E69' for m in models_2025},
    'nnU-Net': '#808080'
}

# ===============================================================================================================
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(11, 7))
# Define legend handles
legend_handles = [
    mpatches.Patch(color='#00BDD6', label='GPT 4o Models'),
    mpatches.Patch(color='#FF5E69', label='GPT o4-mini Models'),
    mpatches.Patch(color='#808080', label='nnU-Net')
]
# Validation subplot
val_list = [validation_scores[m] for m in plot_order_val]
bp = axes[0].boxplot(val_list, patch_artist=True, vert=False,
                     labels=plot_order_val, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order_val):
    patch.set_facecolor(color_map[model])
axes[0].set_title('Brain Tumor Dataset', fontsize=16)
axes[0].set_xlabel('Validation Dice Score', fontsize=14)
axes[0].invert_yaxis()   # highest-median at top
axes[0].set_xlim(0, 1)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[0].grid(True)

# After Validation subplot formatting
for label in axes[0].get_yticklabels():
    if label.get_text() == "nnU-Net":
        label.set_fontweight("bold")

axes[0].legend(handles=legend_handles, loc='upper left', fontsize=11, frameon=True)

# Test subplot
test_list = [test_scores[m] for m in plot_order_test]
bp = axes[1].boxplot(test_list, patch_artist=True, vert=False,
                     labels=plot_order_test, showfliers=False)
for patch, model in zip(bp['boxes'], plot_order_test):
    patch.set_facecolor(color_map[model])
axes[1].set_title('Brain Tumor Dataset', fontsize=16)
axes[1].set_xlabel('Dice Score', fontsize=14)
axes[1].invert_yaxis()
axes[1].set_xlim(0, 1)
axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
axes[1].grid(True)

# After Test subplot formatting
for label in axes[1].get_yticklabels():
    if label.get_text() == "nnU-Net":
        label.set_fontweight("bold")

axes[1].legend(handles=legend_handles, loc='upper left', fontsize=11, frameon=True)

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

fig.suptitle('Model Dice Scores Comparison (Brain Tumor Dataset)', fontsize=22)
plt.tight_layout(rect=[0,0,1,0.96])
plt.subplots_adjust(wspace=0.7)

# Save
save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\plots\\"
plt.savefig(f"{save_path}all_model_runs_dice_scores_combined_BRAIN.png", dpi=600)

# ===============================================================================================================
# Statistical test: Mann–Whitney U test between GPT-4o (2024) and GPT o4-mini (2025)
# ===============================================================================================================

# Helper: compute mean per column for a list of models
# Helper: compute mean per column for a list of models
def compute_group_mean(models, test_scores):
    # Build dict of arrays
    group_dict = {m: test_scores[m] for m in models}
    # Create DataFrame with unequal lengths → fills missing with NaN
    df = pd.DataFrame.from_dict(group_dict, orient="index").transpose()
    # Mean per column, ignoring NaN
    return df.mean(axis=1, skipna=True).values


# Compute mean dice per epoch (column) for each group
mean_2024 = compute_group_mean(models_2024, test_scores)
mean_2025 = compute_group_mean(models_2025, test_scores)

# Run Mann–Whitney U test
u_stat, p_val = mannwhitneyu(mean_2024, mean_2025, alternative='two-sided')

# Print results
print("=== Mann–Whitney U Test (Test Dice Scores, per-epoch means) ===")
print(f"U statistic = {u_stat:.3f}")
print(f"p-value     = {p_val:.6f}")

avg_2024 = np.mean(mean_2024)
avg_2025 = np.mean(mean_2025)

if p_val < 0.05:
    print("Result: Significant difference between groups.")
else:
    print("Result: No significant difference between groups.")

if avg_2024 > avg_2025:
    print(f"GPT 4o (2024) has higher mean Dice ({avg_2024:.4f} vs {avg_2025:.4f})")
elif avg_2025 > avg_2024:
    print(f"GPT o4 mini (2025) has higher mean Dice ({avg_2025:.4f} vs {avg_2024:.4f})")
else:
    print("Both groups have equal mean Dice.")
