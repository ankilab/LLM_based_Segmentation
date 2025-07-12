import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── 1. Paths for 2024 models ──────────────────────────────────────────────────
model_paths_2024 = {
    'Bing Copilot':       {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Bing Microsoft Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Claude 3.5 Sonnet':  {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Claude 3.5 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Copilot':            {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Copilot\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Gemini 1.5 Pro':     {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\Gemini 1.5 pro\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4':              {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT 4o':             {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT 4o\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o1 Preview':     {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\GPT o1 preview\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Llama 3.1 405B':     {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\LLAMA 3.1 405B\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'nnU-Net':            {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
    }
}

# ─── 2. Paths for 2025 models ──────────────────────────────────────────────────
model_paths_2025 = {
    'Claude 4 Sonnet':    {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Claude 4 Sonnet\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'DeepSeek R1':        {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek R1\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'DeepSeek V3':        {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\DeepSeek V3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o3':             {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'GPT o4-mini-high':   {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\GPT o4-mini-high\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Gemini 2.5 pro':     {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Gemini 2.5 Pro\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Grok 3 mini':        {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3 mini Reasoning\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Grok 3':             {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Grok 3\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Llama 4 Maverick':   {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Llama 4 Maverick\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'Mistral Medium 3':   {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\BAGLS output\mask fix\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Mistral Medium 3\out of the box\BAGLS output\mask fix\test_dice_scores.xlsx"
    },
    'Qwen 3_235B':        {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\BAGLS output\validation_dice_scores.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\2025\Qwen 3_235B\out of the box\BAGLS output\test_dice_scores.xlsx"
    },
    'nnU-Net':            {
        'validation': r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\validation_dice_scores_nnUnet_BAGLS.xlsx",
        'test':       r"D:\qy44lyfe\LLM segmentation\Results\nnUnet Baseline\test_dice_scores_nnUnet_BAGLS.xlsx"
    }
}

def load_all_scores(paths_dict):
    vals, tests = [], []
    for p in paths_dict.values():
        v = pd.read_excel(p['validation'], header=None).values.flatten()
        t = pd.read_excel(p['test'],       header=None).values.flatten()
        v = pd.to_numeric(v, errors='coerce')
        t = pd.to_numeric(t, errors='coerce')
        vals.extend(v[(v>=0)&(v<1)])
        tests.extend(t[(t>=0)&(t<1)])
    return vals, tests

# ─── 3. Aggregate 2024 vs 2025 ─────────────────────────────────────────────────
val_2024, test_2024 = load_all_scores(model_paths_2024)
val_2025, test_2025 = load_all_scores(model_paths_2025)

# ─── 4. Plot boxplots ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
fig.suptitle("2024 vs 2025 Dice Scores (BAGLS)", fontsize=12)

data_val  = [val_2024, val_2025]
data_test = [test_2024, test_2025]
labels    = ["2024", "2025"]
colors    = ["#00BDD6", "#FF5E69"]

# Validation subplot
b1 = ax1.boxplot(data_val, patch_artist=True, labels=labels, showfliers=False)
for patch, col in zip(b1['boxes'], colors):
    patch.set_facecolor(col)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax1.set_title("Validation", fontsize=12)
ax1.set_ylabel("Dice Score", fontsize=10)
ax1.tick_params(labelsize=10)
#ax1.legend([b1["boxes"][0], b1["boxes"][1]], labels, loc='upper left')
ax1.grid(True, linestyle="--", linewidth=0.5)

# Test subplot
b2 = ax2.boxplot(data_test, patch_artist=True, labels=labels, showfliers=False)
for patch, col in zip(b2['boxes'], colors):
    patch.set_facecolor(col)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax2.set_title("Test", fontsize=12)
ax2.set_ylabel("Dice Score", fontsize=10)
ax2.tick_params(labelsize=10)
#ax2.legend([b2["boxes"][0], b2["boxes"][1]], labels, loc='upper left')
ax2.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ─── 5. Save figure ────────────────────────────────────────────────────────────
out_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Reason\2024_vs_2025_dice_BAGLS.png"
plt.savefig(out_path, dpi=600)
#plt.show()
