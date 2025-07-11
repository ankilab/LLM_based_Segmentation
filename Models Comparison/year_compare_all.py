import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. File paths ─────────────────────────────────────────────────────────────
train_2024_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_train_losses_BAGLS.xlsx"
val_2024_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_validation_losses_BAGLS.xlsx"
train_2025_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_train_losses_BAGLS.xlsx"
val_2025_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_validation_losses_BAGLS.xlsx"

# ─── 2. Load DataFrames & drop nnU-Net ─────────────────────────────────────────
train_2024 = pd.read_excel(train_2024_path, index_col=0).drop("nnU-Net", errors="ignore").astype(float)
val_2024   = pd.read_excel(val_2024_path,   index_col=0).drop("nnU-Net", errors="ignore").astype(float)
train_2025 = pd.read_excel(train_2025_path, index_col=0).drop("nnU-Net", errors="ignore").astype(float)
val_2025   = pd.read_excel(val_2025_path,   index_col=0).drop("nnU-Net", errors="ignore").astype(float)

# ─── 3. Compute per-epoch mean & std (axis=0) ─────────────────────────────────
def per_epoch_stats(df):
    mean = df.mean(axis=0, skipna=True)
    std  = df.std(axis=0,  skipna=True)
    x = np.arange(1, len(mean) + 1)
    return x, mean.values, std.values

x_t24, y_t24, s_t24 = per_epoch_stats(train_2024)
x_t25, y_t25, s_t25 = per_epoch_stats(train_2025)
x_v24, y_v24, s_v24 = per_epoch_stats(val_2024)
x_v25, y_v25, s_v25 = per_epoch_stats(val_2025)

# Narrower envelope factor
env_factor = 0.5

# ─── 4. Plot #1: Linear y-axis ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Training
ax1.plot(x_t24, y_t24, label="2024 Mean", color="C0")
ax1.fill_between(x_t24,
                 y_t24 - env_factor * s_t24,
                 y_t24 + env_factor * s_t24,
                 color="C0", alpha=0.2)
ax1.plot(x_t25, y_t25, label="2025 Mean", color="C1")
ax1.fill_between(x_t25,
                 y_t25 - env_factor * s_t25,
                 y_t25 + env_factor * s_t25,
                 color="C1", alpha=0.2)
ax1.set_title("Training Loss: 2024 vs 2025")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Average Loss")
ax1.legend()
ax1.grid(True, linestyle="--", linewidth=0.5)

# Validation
ax2.plot(x_v24, y_v24, label="2024 Mean", color="C0")
ax2.fill_between(x_v24,
                 y_v24 - env_factor * s_v24,
                 y_v24 + env_factor * s_v24,
                 color="C0", alpha=0.2)
ax2.plot(x_v25, y_v25, label="2025 Mean", color="C1")
ax2.fill_between(x_v25,
                 y_v25 - env_factor * s_v25,
                 y_v25 + env_factor * s_v25,
                 color="C1", alpha=0.2)
ax2.set_title("Validation Loss: 2024 vs 2025")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Average Loss")
ax2.legend()
ax2.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout()
# Save linear-scale figure
out_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Reason\2024_vs_2025 losses.png"
plt.savefig(out_path, dpi=600)
plt.show()

# ─── 5. Plot #2: Logarithmic y-axis ─────────────────────────────────────────────
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
yticks_custom = [0.001, 0.01, 0.1, 1]

# Training (log)
ax3.plot(x_t24, y_t24, label="2024 Mean", color="C0")
ax3.fill_between(x_t24,
                 y_t24 - env_factor * s_t24,
                 y_t24 + env_factor * s_t24,
                 color="C0", alpha=0.2)
ax3.plot(x_t25, y_t25, label="2025 Mean", color="C1")
ax3.fill_between(x_t25,
                 y_t25 - env_factor * s_t25,
                 y_t25 + env_factor * s_t25,
                 color="C1", alpha=0.2)
ax3.set_yscale("log")
ax3.set_yticks(yticks_custom)
ax3.set_title("Training Loss (log scale)")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Average Loss")
ax3.legend()
ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

# Validation (log)
ax4.plot(x_v24, y_v24, label="2024 Mean", color="C0")
ax4.fill_between(x_v24,
                 y_v24 - env_factor * s_v24,
                 y_v24 + env_factor * s_v24,
                 color="C0", alpha=0.2)
ax4.plot(x_v25, y_v25, label="2025 Mean", color="C1")
ax4.fill_between(x_v25,
                 y_v25 - env_factor * s_v25,
                 y_v25 + env_factor * s_v25,
                 color="C1", alpha=0.2)
ax4.set_yscale("log")
ax4.set_yticks(yticks_custom)
ax4.set_title("Validation Loss (log scale)")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Average Loss")
ax4.legend()
ax4.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
# Save log‐scale figure
plt.savefig(out_path.replace(".png", "_log.png"), dpi=600)
plt.show()
