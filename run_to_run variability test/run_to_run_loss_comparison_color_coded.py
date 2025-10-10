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

save_path = r"D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\plots\loss mean std removed"
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
    nnunet_line = mlines.Line2D([], [], color=col_nnunet, linewidth=2, label="nnU-Net")

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

    axes[0].set_title("Brain Tumor Dataset", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].grid(True)
    axes[0].legend(handles=[model_line, nnunet_line], fontsize=12)
    axes[0].tick_params(axis="both", which="major", labelsize=12)
    axes[0].set_ylim(bottom=0)

    # Validation
    for i, model in enumerate(val_models):
        epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    axes[1].set_title("Brain Tumor Dataset", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].grid(True)
    axes[1].legend(handles=[model_line, nnunet_line], fontsize=12)
    axes[1].tick_params(axis="both", which="major", labelsize=12)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, f"{year}_all_model_losses_BRAIN_linear.png"), dpi=600)
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

    axes[0].set_title("Brain Tumor Dataset", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].set_yscale("log")
    axes[0].set_yticks(yticks_custom)
    axes[0].set_ylim(bottom=1e-3)
    axes[0].grid(True, which="both")
    axes[0].legend(handles=[model_line, nnunet_line], fontsize=12)
    axes[0].tick_params(axis="both", which="major", labelsize=12)

    # Validation (log)
    for i, model in enumerate(val_models):
        epochs = range(1, len(val_losses.iloc[i].dropna()) + 1)
        if model == "nnU-Net":
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_nnunet, linewidth=2)
        else:
            axes[1].plot(epochs, val_losses.iloc[i].dropna(), color=col_model, linewidth=1)

    axes[1].set_title("Brain Tumor Dataset", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].set_yscale("log")
    axes[1].set_yticks(yticks_custom)
    axes[1].set_ylim(bottom=1e-3)
    axes[1].grid(True, which="both")
    axes[1].legend(handles=[model_line, nnunet_line], fontsize=12)
    axes[1].tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, f"{year}_all_model_losses_BRAIN_log.png"), dpi=600)
    plt.close()


# ─── Combined comparison ─────────────────────────────────────────────────────
def plot_combined(train_2024, val_2024, train_2025, val_2025, save_dir, log=False):
    t24_m, t24_l = prepare_loss_data(train_2024)
    v24_m, v24_l = prepare_loss_data(val_2024)
    t25_m, t25_l = prepare_loss_data(train_2025)
    v25_m, v25_l = prepare_loss_data(val_2025)

    c24, c25, cN = "#00BDD6", "#FF5E69", "gray"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scale = "Log Scale" if log else "Normal Scale"
    fig.suptitle(f"Loss Comparison GPT 4o vs GPT o4-mini ({scale})", fontsize=18)

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
        mlines.Line2D([], [], color=c25, linewidth=1, label="GPT o4-mini Models"),
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
