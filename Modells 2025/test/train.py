import os
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = F.binary_cross_entropy_with_logits(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, device, thresh=0.5):
    model.eval()
    running_loss = 0.0
    dice_scores  = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            logits = model(imgs)
            loss   = F.binary_cross_entropy_with_logits(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thresh).astype(np.uint8)
            trues = masks.cpu().numpy().astype(np.uint8)

            # compute Dice per sample in batch
            for p, t in zip(preds, trues):
                p_flat = p.reshape(-1)
                t_flat = t.reshape(-1)
                intersection = np.logical_and(p_flat, t_flat).sum()
                dice = (2.0 * intersection) / (p_flat.sum() + t_flat.sum() + 1e-6)
                dice_scores.append(dice)

    mean_loss = running_loss / len(loader.dataset)
    mean_dice = float(np.mean(dice_scores))
    return mean_loss, mean_dice


def test_epoch(model, loader, device, thresh=0.5):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = (probs >= thresh).astype(np.uint8)
            trues = masks.cpu().numpy().astype(np.uint8)

            for p, t in zip(preds, trues):
                p_flat = p.reshape(-1)
                t_flat = t.reshape(-1)
                intersection = np.logical_and(p_flat, t_flat).sum()
                dice = (2.0 * intersection) / (p_flat.sum() + t_flat.sum() + 1e-6)
                dice_scores.append(dice)

    return float(np.mean(dice_scores))


def save_excel(data, path):
    # ... your existing code for writing Excel ...
    pass


def plot_losses(train_losses, val_losses, save_dir):
    # ... your existing code for plotting loss curves ...
    pass


def plot_predictions(model, loader, device, save_dir, case_indices=None, thresh=0.5):
    import matplotlib.pyplot as plt
    from nibabel import load as load_nii

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # default: first three cases
    if case_indices is None:
        case_indices = list(range(len(loader.dataset)))[:3]

    for idx in case_indices:
        img_path = loader.dataset.images[idx]
        msk_path = loader.dataset.masks[idx]

        img_nii = load_nii(img_path)
        msk_nii = load_nii(msk_path)
        img_vol = img_nii.get_fdata()
        msk_vol = msk_nii.get_fdata()

        # center slice
        z = img_vol.shape[2] // 2
        img_slice = img_vol[:, :, z]
        msk_slice = msk_vol[:, :, z]

        # full-volume inference
        inp = torch.from_numpy(img_vol[None, None]).float().to(device)
        with torch.no_grad():
            logits = model(inp)
            probs  = torch.sigmoid(logits).cpu().numpy()[0, 0]
        pred_slice = (probs[:, :, z] >= thresh).astype(np.uint8)

        # plot high-res, no interpolation
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        for ax in axs: ax.axis("off")

        axs[0].imshow(img_slice.T,  cmap="gray", origin="lower", interpolation="none")
        axs[0].set_title(f"T2 image (slice {z})")

        axs[1].imshow(msk_slice.T, cmap="gray", origin="lower", interpolation="none")
        axs[1].set_title(f"GT mask (slice {z})")

        axs[2].imshow(pred_slice.T, cmap="gray", origin="lower", interpolation="none")
        axs[2].set_title(f"Pred mask (slice {z})")

        plt.tight_layout()
        out_png = os.path.join(save_dir, f"case_{idx:03d}_pred.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
