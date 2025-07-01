import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import UMDDataset
from model import UNet3D
from train import (
    train_epoch, validate_epoch, test_epoch,
    save_excel, plot_losses, plot_predictions
)

def gather_pairs(images_dir, labels_dir):
    import glob, os
    imgs = sorted(glob.glob(os.path.join(images_dir,"*.nii.gz")))
    msks = sorted(glob.glob(os.path.join(labels_dir,"*.nii.gz")))
    mask_map = {os.path.basename(m)[:-len(".nii.gz")]: m for m in msks}
    paired = []
    for im in imgs:
        fn = os.path.basename(im)
        if not fn.endswith("_t2_0000.nii.gz"): continue
        key = fn.replace("_t2_0000.nii.gz","_t2")
        if key in mask_map:
            paired.append((im, mask_map[key]))
    return zip(*paired)

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    im, ms = gather_pairs(args.data_root+"/imagesTr",
                          args.data_root+"/labelsTr")
    n = len(im)
    print(f"Found {n} volumes.")

    idxs = list(range(n))
    tr_i, rst = train_test_split(idxs, test_size=0.2, random_state=args.seed)
    vl, te   = train_test_split(rst, test_size=0.5, random_state=args.seed)

    def sub(lst, idx): return [lst[i] for i in idx]
    tr_ds = UMDDataset(sub(im,tr_i), sub(ms,tr_i), patch_size=args.patch_size, train=True)
    vl_ds = UMDDataset(sub(im,vl), sub(ms,vl), patch_size=args.patch_size, train=False)
    te_ds = UMDDataset(sub(im,te), sub(ms,te), patch_size=args.patch_size, train=False)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    vl_ld = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=1,          shuffle=False, num_workers=2)

    print(f"Train/Val/Test: {len(tr_ds)}/{len(vl_ds)}/{len(te_ds)}")

    # summary on a single patch batch
    try:
        sample, _ = next(iter(tr_ld))
        summary(UNet3D(1,1).to(device),
                input_data=[sample.to(device)],
                col_names=["input_size","output_size","num_params"])
    except Exception as e:
        print("Summary failed:", e)

    model     = UNet3D(1,1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        mode="min", patience=3, factor=0.5)

    train_losses, val_losses, val_dices = [], [], []
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        trl = train_epoch(model, tr_ld, optimizer, device)
        vll, dices = validate_epoch(model, vl_ld, device)
        scheduler.step(vll)

        train_losses.append(trl)
        val_losses.append(vll)
        val_dices.append(dices)

        save_excel([list(range(1,ep+1)), train_losses],
                   os.path.join(args.save_path,"train_losses.xlsx"))
        save_excel([list(range(1,ep+1)), val_losses],
                   os.path.join(args.save_path,"val_losses.xlsx"))
        save_excel(val_dices,
                   os.path.join(args.save_path,"val_dices.xlsx"))

    print(f"\nTraining done in {(time.time()-t0)/60:.2f}min")
    torch.save(model,     os.path.join(args.save_path,"model_full.pth"))
    torch.save(model.state_dict(),
               os.path.join(args.save_path,"model_state.pth"))

    # pick best threshold on val
    best_t = 0.5
    best_d = 0.0
    for t in np.linspace(0.1,0.9,17):
        all_d = [d for epoch in val_dices for d in d]
        mean_d = np.mean([dd for dd in all_d if dd>0])  # exclude zeros
        if mean_d>best_d:
            best_d, best_t = mean_d, t

    print(f"Best val dice {best_d:.4f} @ thresh {best_t:.2f}")

    test_d = test_epoch(model, te_ld, device, best_thresh=best_t)
    save_excel([test_d], os.path.join(args.save_path,"test_dices.xlsx"))

    plot_losses(train_losses, val_losses, args.save_path)
    plot_predictions(model, te_ld, device, args.save_path)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--save_path",   type=str, required=True)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--patch_size",  type=int,   nargs=3, default=(64,64,32))
    args = p.parse_args()
    main(args)
