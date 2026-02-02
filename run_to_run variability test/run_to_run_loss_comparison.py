import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines

# ─── File paths ───────────────────────────────────────────────────────────────
train_2024_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\all_train_losses_GPT 4o runs_BRAIN.xlsx"
val_2024_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\all_val_losses_GPT 4o runs_BRAIN.xlsx"
train_2025_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\all_train_losses_GPT o4 mini high runs_BRAIN.xlsx"
val_2025_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\all_val_losses_GPT o4 mini high runs_BRAIN.xlsx"

save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\plots"
os.makedirs(save_path, exist_ok=True)


# ─── Helper to load and split models/losses ───────────────────────────────────
def prepare_loss_data(path):
    df = pd.read_excel(path, sheet_name=0)
    models = df.iloc[:, 0]  # first column = model names
    losses = df.iloc[:, 1:]  # remaining = per-epoch losses
    return models, losses


# ─── Compute mean, std, and min loss per epoch ────────────────────────────────
def compute_mean_std(models, losses):
    mask = models != "nnU-Net"
    df = losses.loc[mask].astype(float)
    mean = df.mean(axis=0, skipna=True)
    std = df.std(axis=0, skipna=True)
    min_losses = df.min(axis=0, skipna=True)  # per-epoch min loss across models
    return mean, std, min_losses


# ─── Clipping function ────────────────────────────────────────────────────────
def clip_lower(mean, std, min_losses):
    lower = mean - std
    for i in range(len(lower)):
        if lower[i] < min_losses[i]:
            lower[i] = max(lower[i], min_losses[i] * 0.5)
    upper = mean + std
    return lower, upper


# ─── Individual plotting ─────────────────────────────────────────────────────
def plot_models(train_path, val_path, year, save_dir):
    train_models, train_losses = prepare_loss_data(train_path)
    val_models, val_losses = prepare_loss_data(val_path)

    col_model = "#00BDD6" if year == "GPT 4o" else "#FF5E69"
    col_nnunet = "gray"

    model_line = mlines.Line2D([], [], color=col_model, linewidth=1, label=f"{year} Models")
    mean_line = mlines.Line2D([], [], color=col_model, linewidth=2.5, linestyle="--", label=f"{year} Mean ± Std")
    nnunet_line = mlines.Line2D([], [], color=col_nnunet, linewidth=2, label="nnU-Net")

    mean_train, std_train, min_train = compute_mean_std(train_models, train_losses)
    mean_val, std_val, min_val = compute_mean_std(val_models, val_losses)

    # Clip std bounds
    lower_train, upper_train = clip_lower(mean_train, std_train, min_train)
    lower_val, upper_val = clip_lower(mean_val, std_val, min_val)

    # ─ Normal Scale ─
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Loss Comparison ({year} Models)", fontsize=18)

    # Training
    for i, model in enumerate(train_models):
        epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[0].plot(epochs, train_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[0].plot(epochs, train_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    x_train = np.arange(1, len(mean_train) + 1)
    axes[0].plot(x_train, mean_train, color=col_model, linewidth=2.5, linestyle="--")
    axes[0].fill_between(x_train, lower_train, upper_train, color=col_model, alpha=0.2)
    axes[0].set_title("Brain Tumor Dataset", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].grid(True)
    axes[0].legend(handles=[model_line, mean_line, nnunet_line], fontsize=12)
    axes[0].tick_params(axis="both", which="major", labelsize=12)
    axes[0].set_ylim(bottom=0)

    # Validation
    for i, model in enumerate(val_models):
        epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    x_val = np.arange(1, len(mean_val) + 1)
    axes[1].plot(x_val, mean_val, color=col_model, linewidth=2.5, linestyle="--")
    axes[1].fill_between(x_val, lower_val, upper_val, color=col_model, alpha=0.2)
    axes[1].set_title("Brain Tumor Dataset", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].grid(True)
    axes[1].legend(handles=[model_line, mean_line, nnunet_line], fontsize=12)
    axes[1].tick_params(axis="both", which="major", labelsize=12)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, f"{year}_all_model_losses_BRAIN_mean_linear.png"), dpi=600)
    plt.close()

    # ─ Logarithmic Scale ─
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Loss Comparison ({year} Models, Log Scale)", fontsize=18)
    yticks_custom = [0.001, 0.01, 0.1, 1]

    # Training (log)
    for i, model in enumerate(train_models):
        epochs = range(1, len(train_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[0].plot(epochs, train_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[0].plot(epochs, train_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    axes[0].plot(x_train, mean_train, color=col_model, linewidth=2.5, linestyle="--")
    axes[0].fill_between(x_train, lower_train, upper_train, color=col_model, alpha=0.2)
    axes[0].set_title("Brain Tumor Dataset", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].set_yscale("log")
    axes[0].set_yticks(yticks_custom)
    axes[0].set_ylim(bottom=1e-3)
    axes[0].grid(True, which="both")
    axes[0].legend(handles=[model_line, mean_line, nnunet_line], fontsize=12)
    axes[0].tick_params(axis="both", which="major", labelsize=12)

    # Validation (log)
    for i, model in enumerate(val_models):
        epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    axes[1].plot(x_val, mean_val, color=col_model, linewidth=2.5, linestyle="--")
    axes[1].fill_between(x_val, lower_val, upper_val, color=col_model, alpha=0.2)
    axes[1].set_title("Brain Tumor Dataset", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].set_yscale("log")
    axes[1].set_yticks(yticks_custom)
    axes[1].set_ylim(bottom=1e-3)
    axes[1].grid(True, which="both")
    axes[1].legend(handles=[model_line, mean_line, nnunet_line], fontsize=12)
    axes[1].tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, f"{year}_all_model_losses_BRAIN_mean_log.png"), dpi=600)
    plt.close()


# ─── Combined comparison ─────────────────────────────────────────────────────
def plot_combined(train_2024, val_2024, train_2025, val_2025, save_dir, log=False):
    t24_m, t24_l = prepare_loss_data(train_2024)
    v24_m, v24_l = prepare_loss_data(val_2024)
    t25_m, t25_l = prepare_loss_data(train_2025)
    v25_m, v25_l = prepare_loss_data(val_2025)

    c24, c25, cN = "#00BDD6", "#FF5E69", "gray"

    mt24, st24, min_t24 = compute_mean_std(t24_m, t24_l)
    mv24, sv24, min_v24 = compute_mean_std(v24_m, v24_l)
    mt25, st25, min_t25 = compute_mean_std(t25_m, t25_l)
    mv25, sv25, min_v25 = compute_mean_std(v25_m, v25_l)

    # Clip bounds
    lt24, ut24 = clip_lower(mt24, st24, min_t24)
    lt25, ut25 = clip_lower(mt25, st25, min_t25)
    lv24, uv24 = clip_lower(mv24, sv24, min_v24)
    lv25, uv25 = clip_lower(mv25, sv25, min_v25)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scale = "Log Scale" if log else "Normal Scale"
    fig.suptitle(f"Loss Comparison 2024 vs 2025 ({scale})", fontsize=18)

    # TRAIN
    for i, m in enumerate(t24_m):
        e = range(1, len(t24_l.iloc[i].dropna()) + 1)
        if m == "nnU-Net":
            axes[0].plot(e, t24_l.iloc[i].dropna(), color=cN, linewidth=2)
        else:
            axes[0].plot(e, t24_l.iloc[i].dropna(), color=c24, linewidth=1)

    for i, m in enumerate(t25_m):
        e = range(1, len(t25_l.iloc[i].dropna()) + 1)
        if m != "nnU-Net":
            axes[0].plot(e, t25_l.iloc[i].dropna(), color=c25, linewidth=1)

    x24 = np.arange(1, len(mt24) + 1)
    x25 = np.arange(1, len(mt25) + 1)
    axes[0].plot(x24, mt24, color=c24, linewidth=2.5, linestyle="--")
    axes[0].fill_between(x24, lt24, ut24, color=c24, alpha=0.2)
    axes[0].plot(x25, mt25, color=c25, linewidth=2.5, linestyle="--")
    axes[0].fill_between(x25, lt25, ut25, color=c25, alpha=0.2)
    axes[0].set_title("Brain Tumor Dataset", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].grid(True)
    axes[0].tick_params(axis="both", which="major", labelsize=12)
    if log:
        axes[0].set_yscale("log")
        axes[0].set_yticks([0.001, 0.01, 0.1, 1])
        axes[0].set_ylim(bottom=1e-3)
        axes[0].grid(True, which="both")
    else:
        axes[0].set_ylim(bottom=0)

    # VAL
    for i, m in enumerate(v24_m):
        e = range(1, len(v24_l.iloc[i].dropna()) + 1)
        if m == "nnU-Net":
            axes[1].plot(e, v24_l.iloc[i].dropna(), color=cN, linewidth=2)
        else:
            axes[1].plot(e, v24_l.iloc[i].dropna(), color=c24, linewidth=1)

    for i, m in enumerate(v25_m):
        e = range(1, len(v25_l.iloc[i].dropna()) + 1)
        if m != "nnU-Net":
            axes[1].plot(e, v25_l.iloc[i].dropna(), color=c25, linewidth=1)

    x24v = np.arange(1, len(mv24) + 1)
    x25v = np.arange(1, len(mv25) + 1)
    axes[1].plot(x24v, mv24, color=c24, linewidth=2.5, linestyle="--")
    axes[1].fill_between(x24v, lv24, uv24, color=c24, alpha=0.2)
    axes[1].plot(x25v, mv25, color=c25, linewidth=2.5, linestyle="--")
    axes[1].fill_between(x25v, lv25, uv25, color=c25, alpha=0.2)
    axes[1].set_title("Brain Tumor Dataset", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].grid(True)
    axes[1].tick_params(axis="both", which="major", labelsize=12)
    if log:
        axes[1].set_yscale("log")
        axes[1].set_yticks([0.001, 0.01, 0.1, 1])
        axes[1].set_ylim(bottom=1e-3)
        axes[1].grid(True, which="both")
    else:
        axes[1].set_ylim(bottom=0)

    # Shared legend handles
    handles = [
        mlines.Line2D([], [], color=c24, linewidth=1, label="GPT 4o Models"),
        mlines.Line2D([], [], color=c24, linewidth=2.5, linestyle="--", label="GPT 4o Mean ± Std"),
        mlines.Line2D([], [], color=c25, linewidth=1, label="GPT o4-mini Models"),
        mlines.Line2D([], [], color=c25, linewidth=2.5, linestyle="--", label="GPT o4-mini Mean ± Std"),
        mlines.Line2D([], [], color=cN, linewidth=2, label="nnU-Net"),
    ]
    axes[0].legend(handles=handles, fontsize=12)
    axes[1].legend(handles=handles, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = "combined_log.png" if log else "combined_normal.png"
    plt.savefig(os.path.join(save_dir, f"GPT 4o_vs_GPT o4 mini_all_model_losses_BRAIN_{fname}"), dpi=600)
    plt.close()


# ─── Run ──────────────────────────────────────────────────────────────────────
plot_models(train_2024_path, val_2024_path, "GPT 4o", save_path)
plot_models(train_2025_path, val_2025_path, "GPT o4-mini", save_path)
plot_combined(train_2024_path, val_2024_path, train_2025_path, val_2025_path, save_path, log=False)
plot_combined(train_2024_path, val_2024_path, train_2025_path, val_2025_path, save_path, log=True)
