# unet_segmentation/train.py
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


EPS = 1e-7  # to avoid div-by-zero


def dice_coeff(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred_logits : (N,1,H,W) raw logits
    target      : (N,1,H,W) 0/1
    Calculates Dice for each sample, returns Tensor(N,)
    """
    pred = torch.sigmoid(pred_logits)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + EPS) / (union + EPS)
    return dice


# def _epoch_template(
#     loader: DataLoader,
#     model: torch.nn.Module,
#     criterion,
#     device: torch.device,
#     phase: str = "train",
#     optimiser: torch.optim.Optimizer | None = None,
# ) -> Tuple[float, List[float]]:
#     assert phase in {"train", "val", "test"}
#     is_train = phase == "train"
#     model.train(is_train)
#
#     running_loss = 0.0
#     dice_scores: List[float] = []
#
#     loop = tqdm(loader, desc=f"{phase:>5}", leave=False)
#
#     for images, masks, _ in loop:
#         images, masks = images.to(device), masks.to(device)
#         optimiser.zero_grad() if is_train else None
#
#         logits = model(images)
#         loss = criterion(logits, masks)
#         running_loss += loss.item() * images.size(0)
#
#         if is_train:
#             loss.backward()
#             optimiser.step()
#
#         if phase in {"val", "test"}:
#             dice = dice_coeff(logits, masks)      # (N,)
#             dice_scores.extend(dice.detach().cpu().tolist())
#             loop.set_postfix(loss=loss.item(), dice=sum(dice_scores) / len(dice_scores))
#         else:
#             loop.set_postfix(loss=loss.item())
#
#     epoch_loss = running_loss / len(loader.dataset)
#     return epoch_loss, dice_scores

def _epoch_template(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    device: torch.device,
    phase: str = "train",
    optimiser: torch.optim.Optimizer | None = None,
) -> Tuple[float, List[float]]:
    """
    Returns:
        epoch_loss          – average BCE-with-logits loss over *samples*
        batch_dice_scores   – list length == # batches
                              (each item is the mean Dice of that batch)
    """
    assert phase in {"train", "val", "test"}
    is_train = phase == "train"
    model.train(is_train)

    running_loss = 0.0
    batch_dice_scores: List[float] = []

    loop = tqdm(loader, desc=f"{phase:>5}", leave=False)

    for images, masks, _ in loop:
        images, masks = images.to(device), masks.to(device)
        if is_train:
            optimiser.zero_grad()

        logits = model(images)
        loss   = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

        if is_train:
            loss.backward()
            optimiser.step()

        # ── NEW: store *batch-mean* Dice instead of per-sample ────────────
        if phase in {"val", "test"}:
            dice_mean = dice_coeff(logits, masks).mean().item()  # scalar
            batch_dice_scores.append(dice_mean)
            loop.set_postfix(loss=loss.item(), dice=dice_mean)
        else:
            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, batch_dice_scores

def save_vector_as_excel(values: List[float], excel_path: Path, header: str = "loss") -> None:
    """
    values -> [v0, v1, ...]  produces two-row dataframe:
    Epoch | 1 | 2 | ...
    header| v0| v1| ...
    """
    df = pd.DataFrame([range(1, len(values) + 1), values])
    df.index = ["Epoch", header]
    df.to_excel(excel_path, header=False)


def save_2dlist_as_excel(rows: List[List[float]], excel_path: Path) -> None:
    """
    rows[e][b] -> epoch e, batch b
    Variable length rows are padded with NaN.
    """
    max_len = max(len(r) for r in rows)
    padded = [r + [float("nan")] * (max_len - len(r)) for r in rows]
    df = pd.DataFrame(padded, index=[f"epoch_{i+1}" for i in range(len(rows))])
    df.to_excel(excel_path, index=True)


def plot_losses(train_losses: List[float], val_losses: List[float], save_path: Path) -> None:
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 5,
) -> None:
    import random
    from torchvision.utils import make_grid

    model.eval()
    samples = random.sample(range(len(dataloader.dataset)), num_samples)

    imgs, gts, preds, titles = [], [], [], []
    for idx in samples:
        image, mask, name = dataloader.dataset[idx]
        with torch.no_grad():
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu()
            pred = (pred > 0.5).float()

        imgs.append(image[0])   # remove channel dim
        gts.append(mask[0])
        preds.append(pred)
        titles.append(name)

    # Plot 5×3 grid
    plt.figure(figsize=(9, 3 * num_samples))
    for i in range(num_samples):
        for j, mat in enumerate([imgs, gts, preds]):  # image, gt, pred
            ax = plt.subplot(num_samples, 3, i * 3 + j + 1)
            ax.imshow(mat[i], cmap="gray")
            if i == 0:
                ax.set_title(["Input", "Ground Truth", "Prediction"][j])
            ax.set_xlabel(titles[i])
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_validate(
    model,
    train_loader,
    val_loader,
    optimiser,
    criterion,
    device,
    epochs: int,
    save_dir: Path,
) -> Tuple[List[float], List[float]]:
    save_dir.mkdir(parents=True, exist_ok=True)
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_dice_rows: List[List[float]] = []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        t_loss, _ = _epoch_template(train_loader, model, criterion, device, phase="train", optimiser=optimiser)
        v_loss, v_dice = _epoch_template(val_loader, model, criterion, device, phase="val")

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_dice_rows.append(v_dice)

        print(f"  train_loss={t_loss:.4f}   val_loss={v_loss:.4f}   val_dice_mean={sum(v_dice)/len(v_dice):.4f}")

        # persist intermediate metrics each epoch
        save_vector_as_excel(train_losses, save_dir / "train_losses.xlsx", header="train_loss")
        save_vector_as_excel(val_losses,   save_dir / "val_losses.xlsx",   header="val_loss")
        save_2dlist_as_excel(val_dice_rows, save_dir / "validation_dice_scores.xlsx")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # final plot
    plot_losses(train_losses, val_losses, save_dir / "loss_curve.png")
    return train_losses, val_losses


def test_phase(model, test_loader, criterion, device, save_dir: Path):
    print("\nTesting…")
    test_loss, test_dice = _epoch_template(test_loader, model, criterion, device, phase="test")
    print(f"Test loss={test_loss:.4f}   Dice={sum(test_dice)/len(test_dice):.4f}")

    save_2dlist_as_excel([test_dice], save_dir / "test_dice_scores.xlsx")
    visualize_predictions(model, test_loader, device, save_dir / "sample_predictions.png")
