# ###################################################
# code 1. uncoment section for logarithmic and normal scaled plots
#####################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ─── Helper: Load Excel with comma decimal separator ───────────────────────────
def load_excel(path):
    df = pd.read_excel(path, index_col=0, dtype=str)  # read as strings
    # Convert comma-decimal strings into floats
    df = df.applymap(lambda x: float(x.replace(",", ".").strip()) if pd.notnull(x) else np.nan)
    return df

# ─── 1. File paths ─────────────────────────────────────────────────────────────
train_2024_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_train_losses_myoma.xlsx"
val_2024_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_validation_losses_myoma.xlsx"
train_2025_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_train_losses_myoma.xlsx"
val_2025_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_validation_losses_myoma.xlsx"

# ─── 2. Load DataFrames & keep nnU-Net separately ──────────────────────────────
train_2024_full = load_excel(train_2024_path)
val_2024_full   = load_excel(val_2024_path)
train_2025_full = load_excel(train_2025_path)
val_2025_full   = load_excel(val_2025_path)

train_nnunet, val_nnunet = None, None
if "nnU-Net" in train_2024_full.index:
    train_nnunet = train_2024_full.loc["nnU-Net"].values
    train_2024 = train_2024_full.drop("nnU-Net")
    val_nnunet = val_2024_full.loc["nnU-Net"].values
    val_2024   = val_2024_full.drop("nnU-Net")
else:
    train_2024, val_2024 = train_2024_full, val_2024_full

train_2025 = train_2025_full.drop("nnU-Net", errors="ignore")
val_2025   = val_2025_full.drop("nnU-Net", errors="ignore")

# ─── 3. Debug check: print min and max ─────────────────────────────────────────
for name, df in {
    "train_2024": train_2024, "val_2024": val_2024,
    "train_2025": train_2025, "val_2025": val_2025
}.items():
    print(f"{name}: min={df.min().min():.6f}, max={df.max().max():.6f}")

if train_nnunet is not None:
    print(f"nnU-Net train: min={np.nanmin(train_nnunet):.6f}, max={np.nanmax(train_nnunet):.6f}")
if val_nnunet is not None:
    print(f"nnU-Net val:   min={np.nanmin(val_nnunet):.6f}, max={np.nanmax(val_nnunet):.6f}")

# ─── 4. Compute per-epoch mean & std ──────────────────────────────────────────
def per_epoch_stats(df):
    mean = df.mean(axis=0, skipna=True)
    std  = df.std(axis=0,  skipna=True)
    std  = np.clip(std, 0, None)
    x = np.arange(1, len(mean) + 1)
    return x, mean.values, std.values

x_t24, y_t24, s_t24 = per_epoch_stats(train_2024)
x_t25, y_t25, s_t25 = per_epoch_stats(train_2025)
x_v24, y_v24, s_v24 = per_epoch_stats(val_2024)
x_v25, y_v25, s_v25 = per_epoch_stats(val_2025)

# ─── 5. Get max epochs & max value ────────────────────────────────────────────
all_lengths = [
    len(train_2024.columns), len(val_2024.columns),
    len(train_2025.columns), len(val_2025.columns),
]
all_values = [
    train_2024.values, val_2024.values,
    train_2025.values, val_2025.values,
]
if train_nnunet is not None:
    all_lengths.append(len(train_nnunet))
    all_values.append(train_nnunet)
if val_nnunet is not None:
    all_lengths.append(len(val_nnunet))
    all_values.append(val_nnunet)

max_epochs = max(all_lengths)
max_value = np.nanmax(np.concatenate([np.ravel(v) for v in all_values if v is not None]))

# narrower envelope factor
env_factor = 0.5

# colors
c24 = "#00BDD6"   # strong blue
c25 = "#FF5E69"   # strong red
pale_24 = "#99e6f2"   # pale blue
pale_25 = "#ffb3b8"   # pale red
c_nnunet = "gray"

# ─── 6. Function to make plots ────────────────────────────────────────────────
def make_plot(scale="log", out_file="plot.png"):
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Train Val Loss comparison Models 2024 vs 2025", fontsize=14)

    for ax in (ax3, ax4):
        ax.set_xlim(0, max_epochs)
        ax.tick_params(labelsize=12)
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-3, top=max_value * 1.05)
            # Major ticks only at powers of 10
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
            # Minor ticks (unlabeled small ticks)
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
        else:
            ax.set_ylim(0, max_value * 1.05)

    # Training
    for model in train_2024.index:
        ax3.plot(np.arange(1, len(train_2024.loc[model]) + 1),
                 train_2024.loc[model].values, color=pale_24, linewidth=1, alpha=0.6)
    for model in train_2025.index:
        ax3.plot(np.arange(1, len(train_2025.loc[model]) + 1),
                 train_2025.loc[model].values, color=pale_25, linewidth=1, alpha=0.6)

    ax3.plot(x_t24, y_t24, label="2024 Models Mean", color=c24, linewidth=2)
    ax3.fill_between(x_t24, y_t24 - env_factor * s_t24, y_t24 + env_factor * s_t24, color=c24, alpha=0.2)
    ax3.plot(x_t25, y_t25, label="2025 Models Mean", color=c25, linewidth=2)
    ax3.fill_between(x_t25, y_t25 - env_factor * s_t25, y_t25 + env_factor * s_t25, color=c25, alpha=0.2)

    if train_nnunet is not None:
        ax3.plot(np.arange(1, len(train_nnunet) + 1), train_nnunet, label="nnU-Net", color=c_nnunet, linewidth=2)

    ax3.set_title(f"Uterine Myoma Dataset", fontsize=16)
    ax3.set_xlabel("Epoch", fontsize=14)
    ax3.set_ylabel("Training Loss", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Validation
    for model in val_2024.index:
        ax4.plot(np.arange(1, len(val_2024.loc[model]) + 1),
                 val_2024.loc[model].values, color=pale_24, linewidth=1, alpha=0.6)
    for model in val_2025.index:
        ax4.plot(np.arange(1, len(val_2025.loc[model]) + 1),
                 val_2025.loc[model].values, color=pale_25, linewidth=1, alpha=0.6)

    ax4.plot(x_v24, y_v24, label="2024 Models Mean", color=c24, linewidth=2)
    ax4.fill_between(x_v24, y_v24 - env_factor * s_v24, y_v24 + env_factor * s_v24, color=c24, alpha=0.2)
    ax4.plot(x_v25, y_v25, label="2025 Models Mean", color=c25, linewidth=2)
    ax4.fill_between(x_v25, y_v25 - env_factor * s_v25, y_v25 + env_factor * s_v25, color=c25, alpha=0.2)

    if val_nnunet is not None:
        ax4.plot(np.arange(1, len(val_nnunet) + 1), val_nnunet, label="nnU-Net", color=c_nnunet, linewidth=2)

    ax4.set_title(f"Uterine Myoma Dataset", fontsize=16)
    ax4.set_xlabel("Epoch", fontsize=14)
    ax4.set_ylabel("Validation Loss", fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_file, dpi=600)
    plt.close(fig)

# ─── 7. Make both log and linear plots ─────────────────────────────────────────
out_dir = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Reason\show lines"

make_plot(scale="log", out_file=out_dir + r"\2024_vs_2025_losses_myoma_log.png")
make_plot(scale="linear", out_file=out_dir + r"\2024_vs_2025_losses_myoma_linear.png")


#######################################################
# # code 2. : uncomment if normal scaled loss plots
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ─── 1. File paths ─────────────────────────────────────────────────────────────
# train_2024_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_train_losses_Retina.xlsx"
# val_2024_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2024\all_validation_losses_Retina.xlsx"
# train_2025_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_train_losses_Retina.xlsx"
# val_2025_path   = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\models 2025\all_validation_losses_Retina.xlsx"
#
# # ─── 2. Load DataFrames (keep nnU-Net separately) ──────────────────────────────
# train_2024_full = pd.read_excel(train_2024_path, index_col=0).astype(float)
# val_2024_full   = pd.read_excel(val_2024_path,   index_col=0).astype(float)
# train_2025_full = pd.read_excel(train_2025_path, index_col=0).astype(float)
# val_2025_full   = pd.read_excel(val_2025_path,   index_col=0).astype(float)
#
# # extract nnU-Net if exists
# nnunet_train = None
# nnunet_val = None
# if "nnU-Net" in train_2024_full.index:
#     nnunet_train = train_2024_full.loc["nnU-Net"].values
#     nnunet_val   = val_2024_full.loc["nnU-Net"].values
#
# # drop nnU-Net for mean/std
# train_2024 = train_2024_full.drop("nnU-Net", errors="ignore")
# val_2024   = val_2024_full.drop("nnU-Net", errors="ignore")
# train_2025 = train_2025_full.drop("nnU-Net", errors="ignore")
# val_2025   = val_2025_full.drop("nnU-Net", errors="ignore")
#
# # ─── 3. Compute per-epoch mean & std ───────────────────────────────────────────
# def per_epoch_stats(df):
#     mean = df.mean(axis=0, skipna=True)
#     std  = df.std(axis=0,  skipna=True)
#     std  = np.clip(std, 0, None)
#     x = np.arange(1, len(mean) + 1)
#     return x, mean.values, std.values
#
# # get stats
# x_t24, y_t24, s_t24 = per_epoch_stats(train_2024)
# x_t25, y_t25, s_t25 = per_epoch_stats(train_2025)
# x_v24, y_v24, s_v24 = per_epoch_stats(val_2024)
# x_v25, y_v25, s_v25 = per_epoch_stats(val_2025)
#
# env_factor = 0.5
# c24 = "#00BDD6"
# c25 = "#FF5E69"
#
# # pale colors for individuals
# pale_c24 = (0, 189/255, 214/255, 0.3)  # rgba version with alpha
# pale_c25 = (1, 94/255, 105/255, 0.3)
#
# # ─── 4. Determine max number of epochs ─────────────────────────────────────────
# max_epochs = max(
#     train_2024_full.shape[1],
#     val_2024_full.shape[1],
#     train_2025_full.shape[1],
#     val_2025_full.shape[1],
#     len(nnunet_train) if nnunet_train is not None else 0,
#     len(nnunet_val) if nnunet_val is not None else 0
# )
#
# # set ~10 tick marks
# step = max(1, max_epochs // 10)
# x_ticks = list(range(0, max_epochs + 1, step))
#
# # ─── 5. Plot Linear y-axis ─────────────────────────────────────────────────────
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
# fig.suptitle("Average Loss comparison Models 2024 vs 2025, Retina Dataset", fontsize=14)
#
# for ax in (ax1, ax2):
#     ax.set_xticks(x_ticks)
#     ax.set_xlim(0, max_epochs)
#     ax.tick_params(labelsize=12)
#
# # ─ Training ───────────────────────────────────────────────────────────────────
# # plot individual models (2024 pale blue, 2025 pale red)
# for _, row in train_2024.iterrows():
#     ax1.plot(np.arange(1, len(row)+1), row.values, color=pale_c24, linewidth=1)
# for _, row in train_2025.iterrows():
#     ax1.plot(np.arange(1, len(row)+1), row.values, color=pale_c25, linewidth=1)
#
# # plot means
# ax1.plot(x_t24, y_t24, label="2024 Models Mean", color=c24)
# ax1.fill_between(x_t24, y_t24 - env_factor*s_t24, y_t24 + env_factor*s_t24, color=c24, alpha=0.2)
# ax1.plot(x_t25, y_t25, label="2025 Models Mean", color=c25)
# ax1.fill_between(x_t25, y_t25 - env_factor*s_t25, y_t25 + env_factor*s_t25, color=c25, alpha=0.2)
#
# # plot nnU-Net baseline (if available)
# if nnunet_train is not None:
#     ax1.plot(np.arange(1, len(nnunet_train)+1), nnunet_train, color="gray", linewidth=2, label="nnU-Net")
#
# ax1.set_title("Average Training Loss per Epoch", fontsize=14)
# ax1.set_xlabel("Epoch", fontsize=12)
# ax1.set_ylabel("Average Loss across Models", fontsize=12)
# ax1.legend(fontsize=11)
# ax1.grid(True, linestyle="--", linewidth=0.5)
#
# # ─ Validation ──────────────────────────────────────────────────────────────────
# for _, row in val_2024.iterrows():
#     ax2.plot(np.arange(1, len(row)+1), row.values, color=pale_c24, linewidth=1)
# for _, row in val_2025.iterrows():
#     ax2.plot(np.arange(1, len(row)+1), row.values, color=pale_c25, linewidth=1)
#
# ax2.plot(x_v24, y_v24, label="2024 Models Mean", color=c24)
# ax2.fill_between(x_v24, y_v24 - env_factor*s_v24, y_v24 + env_factor*s_v24, color=c24, alpha=0.2)
# ax2.plot(x_v25, y_v25, label="2025 Models Mean", color=c25)
# ax2.fill_between(x_v25, y_v25 - env_factor*s_v25, y_v25 + env_factor*s_v25, color=c25, alpha=0.2)
#
# if nnunet_val is not None:
#     ax2.plot(np.arange(1, len(nnunet_val)+1), nnunet_val, color="gray", linewidth=2, label="nnU-Net")
#
# ax2.set_title("Average Validation Loss per Epoch", fontsize=14)
# ax2.set_xlabel("Epoch", fontsize=12)
# ax2.set_ylabel("Average Loss across Models", fontsize=12)
# ax2.legend(fontsize=11)
# ax2.grid(True, linestyle="--", linewidth=0.5)
#
# plt.tight_layout(rect=[0, 0, 1, 0.92])
#
# out_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Reason\show lines\2024_vs_2025_losses_Retina_test.png"
# plt.savefig(out_path, dpi=600)
# # plt.show()


