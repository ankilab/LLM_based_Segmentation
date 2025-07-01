import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Losses ----------
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        num  = 2*(probs*target).sum(dim=(1,2,3,4)) + self.smooth
        den  = (probs+target).sum(dim=(1,2,3,4)) + self.smooth
        dice = (num/den).mean()
        return 1 - dice

bce_loss  = torch.nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()
def combined_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)

# ---------- Metrics ----------
def dice_coeff_batch(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy() > threshold
    t = target.cpu().numpy() > 0.5
    dices = []
    for p, gt in zip(probs, t):
        intersect = (p & gt).sum()
        union     = p.sum() + gt.sum()
        dices.append((2*intersect)/(union+1e-6) if union>0 else 1.0)
    return dices

# ---------- Epoch routines ----------
def train_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pbar.set_postfix(loss=f"{running/(pbar.n):.4f}")
    return running/len(loader)

def validate_epoch(model, loader, device):
    model.eval()
    running = 0.0
    all_dices = []
    pbar = tqdm(loader, desc="Val  ", leave=False)
    with torch.no_grad():
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = combined_loss(logits, masks)
            running += loss.item()
            dices = dice_coeff_batch(logits, masks)
            all_dices.append(dices)
            pbar.set_postfix(loss=f"{running/(pbar.n):.4f}", dice=f"{np.mean(dices):.4f}")
    return running/len(loader), all_dices

def test_epoch(model, loader, device, best_thresh=0.5):
    model.eval()
    all_dices = []
    pbar = tqdm(loader, desc="Test ", leave=False)
    with torch.no_grad():
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dices = dice_coeff_batch(logits, masks, threshold=best_thresh)
            all_dices.append(dices)
            pbar.set_postfix(dice=f"{np.mean(dices):.4f}")
    # flatten
    return [d for batch in all_dices for d in batch]

# ---------- I/O & plotting ----------
def save_excel(data_rows, filepath):
    df = pd.DataFrame(data_rows)
    df.to_excel(filepath, header=False, index=False)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def plot_predictions(model, loader, device, save_path):
    import random, numpy as np
    fig, axes = plt.subplots(5,3,figsize=(9,15))
    indices = random.sample(range(len(loader.dataset)), 5)
    for i, idx in enumerate(indices):
        img, gt = loader.dataset[idx]
        img_b = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_b)).cpu()[0,0]
        img_np = img[0].numpy()
        gt_np  = gt[0].numpy()
        pr_np  = (pred.numpy()>0.5).astype(np.float32)

        axes[i,0].imshow(img_np[img_np.shape[0]//2], cmap="gray")
        axes[i,0].set_title(f"Input (idx={idx})")
        axes[i,1].imshow(gt_np[gt_np.shape[0]//2], cmap="gray")
        axes[i,1].set_title("GT")
        axes[i,2].imshow(pr_np[pr_np.shape[0]//2], cmap="gray")
        axes[i,2].set_title("Pred")
        for j in range(3): axes[i,j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions.png"))
    plt.close(fig)
