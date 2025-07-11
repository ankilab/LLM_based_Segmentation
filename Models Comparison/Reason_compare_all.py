import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. File paths ─────────────────────────────────────────────────────────────
train_2024_path = r"/Results/Models Comparison/models 2024/all_train_losses_myoma.xlsx"
val_2024_path   = r"/Results/Models Comparison/models 2024/all_validation_losses_myoma.xlsx"
train_2025_path = r"/Results/Models Comparison/models 2025/all_train_losses_myoma.xlsx"
val_2025_path   = r"/Results/Models Comparison/models 2025/all_validation_losses_myoma.xlsx"

# ─── 2. Load DataFrames & drop nnU-Net ─────────────────────────────────────────
train_2024 = pd.read_excel(train_2024_path, index_col=0)\
               .drop("nnU-Net", errors="ignore").astype(float)
val_2024   = pd.read_excel(val_2024_path,   index_col=0)\
               .drop("nnU-Net", errors="ignore").astype(float)
train_2025 = pd.read_excel(train_2025_path, index_col=0)\
               .drop("nnU-Net", errors="ignore").astype(float)
val_2025   = pd.read_excel(val_2025_path,   index_col=0)\
               .drop("nnU-Net", errors="ignore").astype(float)

# ─── 3. Define reasoning vs non‐reasoning model lists ──────────────────────────
reasoning_2025 = [
    "Claude 4 Sonnet",
    "DeepSeek R1",
    "GPT o3",
    "GPT o4-mini-high",
    "Gemini 2.5 pro",
    "Grok 3 mini",
    "Llama 4 Maverick",
    "Mistral Medium 3",
    "Qwen 3_235B",
]

reasoning_2024 = [
    "GPT o1 Preview",
]

nonreasoning_2025 = [
    "DeepSeek V3",
    "Grok 3",
]

nonreasoning_2024 = [
    "Bing Microsoft Copilot",
    "Claude 3.5 Sonnet",
    "Copilot",
    "Gemini 1.5 Pro",
    "GPT 4o",
    "GPT4",
    "Llama 3.1 405B",
]

# ─── 4. Build combined DataFrames ───────────────────────────────────────────────
train_reasoning = pd.concat([
    train_2024.loc[reasoning_2024],
    train_2025.loc[reasoning_2025]
], axis=0)

train_nonreason = pd.concat([
    train_2024.loc[nonreasoning_2024],
    train_2025.loc[nonreasoning_2025]
], axis=0)

val_reasoning = pd.concat([
    val_2024.loc[reasoning_2024],
    val_2025.loc[reasoning_2025]
], axis=0)

val_nonreason = pd.concat([
    val_2024.loc[nonreasoning_2024],
    val_2025.loc[nonreasoning_2025]
], axis=0)

# ─── 5. Compute per‐epoch mean & std (axis=0), clip negative std to zero ─────
def per_epoch_stats(df):
    mean = df.mean(axis=0, skipna=True)
    std  = df.std(axis=0,  skipna=True)
    std  = np.clip(std, 0, None)
    x = np.arange(1, len(mean) + 1)
    return x, mean.values, std.values

xr, yr, sr = per_epoch_stats(train_reasoning)
xnr, ynr, snr = per_epoch_stats(train_nonreason)
xvr, yvr, svr = per_epoch_stats(val_reasoning)
xvnr, yvnr, svnr = per_epoch_stats(val_nonreason)

env_factor = 0.5
# swap colors: reasoning = red-ish, nonreasoning = blue-ish
c_r = "#FF5E69"
c_nr = "#00BDD6"

# ─── 6. Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
fig.suptitle("Average Loss comparison: Reasoning vs Non‐Reasoning Models, Myoma Dataset", fontsize=14)

# shared x ticks 0–20, whole numbers
xticks = list(range(0, 21, 3))
for ax in (ax1, ax2):
    ax.set_xlim(0, 20)
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=12)

# Training subplot
ax1.plot(xr, yr,   label="Reasoning Models",     color=c_r)
ax1.fill_between(xr, yr - env_factor*sr, yr + env_factor*sr, color=c_r, alpha=0.2)
ax1.plot(xnr, ynr, label="Non‐Reasoning Models", color=c_nr)
ax1.fill_between(xnr, ynr - env_factor*snr, ynr + env_factor*snr, color=c_nr, alpha=0.2)
ax1.set_title("Average Training Loss per Epoch", fontsize=14)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Average Loss across Models", fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle="--", linewidth=0.5)

# Validation subplot
ax2.plot(xvr, yvr,   label="Reasoning Models",     color=c_r)
ax2.fill_between(xvr, yvr - env_factor*svr, yvr + env_factor*svr, color=c_r, alpha=0.2)
ax2.plot(xvnr, yvnr, label="Non‐Reasoning Models", color=c_nr)
ax2.fill_between(xvnr, yvnr - env_factor*svnr, yvnr + env_factor*svnr, color=c_nr, alpha=0.2)
ax2.set_title("Average Validation Loss per Epoch", fontsize=14)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Average Loss across Models", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# ─── 7. Save figure ────────────────────────────────────────────────────────────
out_path = r"/Results/Models Comparison/Reason/Reasoning vs non reasoning losses_Myoma.png"
plt.savefig(out_path, dpi=600)
#plt.show()
