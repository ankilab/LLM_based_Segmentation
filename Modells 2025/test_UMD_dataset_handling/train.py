# train.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, count = 0., 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); count += 1
    return total_loss / count

def validate_epoch(model, loader, device):
    model.eval()
    total_loss, total_dice, count = 0., 0., 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            total_loss += loss.item()
            # flatten and threshold
            pred = (torch.sigmoid(logits)>0.5).cpu().numpy().ravel()
            true = masks.cpu().numpy().ravel()
            total_dice += jaccard_score(true, pred, average='binary', zero_division=0)
            count += 1
    return total_loss/count, total_dice/count

def plot_predictions(model, dataset, device, save_path, n_plots=4):
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    figs = []
    with torch.no_grad():
        for i in range(min(n_plots, len(dataset))):
            img, msk = dataset[i]
            vol = img.numpy()[0]
            z = vol.shape[2]//2
            slice_img = vol[:, :, z].T
            slice_gt  = msk.numpy()[0, :, :, z].T

            inp = img[None].to(device)
            out = torch.sigmoid(model(inp))[0,0].cpu().numpy()
            slice_pr = (out[:, :, z]>0.5).T

            # plot
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            axes[0].imshow(slice_img, cmap='gray');      axes[0].set_title("T2")
            axes[1].imshow(slice_gt,  cmap='gray');      axes[1].set_title("GT Myoma")
            axes[2].imshow(slice_pr, cmap='gray');      axes[2].set_title("Pred Myoma")
            for ax in axes: ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"pred_{i:02d}.png"), dpi=150)
            plt.close(fig)
