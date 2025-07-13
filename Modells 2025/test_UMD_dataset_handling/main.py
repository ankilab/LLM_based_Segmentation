# main.py
import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import UMDDataset
from model   import UNet3D
from train   import train_epoch, validate_epoch

def get_file_list(root):
    imgs = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
    msks = []
    for p in imgs:
        fn = os.path.basename(p).replace("_t2_0000.nii.gz", "_t2.nii.gz")
        msks.append(os.path.join(root, "labelsTr", fn))
    return imgs, msks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs, msks = get_file_list(args.data_root)
    n = len(imgs)
    n_train = int(0.8*n)
    n_val   = int(0.1*n)
    n_test  = n - n_train - n_val

    all_pairs = list(zip(imgs, msks))
    train_ds, val_ds, test_ds = random_split(all_pairs, [n_train, n_val, n_test])

    tr_imgs, tr_msks = zip(*train_ds)
    vl_imgs, vl_msks = zip(*val_ds)
    te_imgs, te_msks = zip(*test_ds)

    tr_loader = DataLoader(
        UMDDataset(tr_imgs, tr_msks, patch_size=(64,64,32), train=True),
        batch_size=args.batch_size, shuffle=True,  num_workers=4
    )
    vl_loader = DataLoader(
        UMDDataset(vl_imgs, vl_msks, patch_size=(64,64,32), train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    te_loader = DataLoader(
        UMDDataset(te_imgs, te_msks, patch_size=(64,64,32), train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = UNet3D(1,1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss":[], "val_loss":[], "val_dice":[]}

    for ep in range(args.epochs):
        trl = train_epoch(model, tr_loader, optim, device)
        vll, vld = validate_epoch(model, vl_loader, device)
        print(f"Epoch {ep+1}/{args.epochs}"
              f"  train_loss={trl:.4f}"
              f"  val_loss={vll:.4f}"
              f"  val_dice={vld:.4f}")
        history["train_loss"].append(trl)
        history["val_loss"].append(vll)
        history["val_dice"].append(vld)

    tll, tld = validate_epoch(model, te_loader, device)
    print(f"\nTest  loss={tll:.4f}  meanDice={tld:.4f}")

    # plot one example (center slice of a test volume)
    import nibabel as nib
    case_img = te_imgs[0]
    case_msk = te_msks[0]
    vol_i    = nib.load(case_img).get_fdata().astype(np.float32)
    vol_m    = nib.load(case_msk).get_fdata()
    vol_m    = (vol_m==3).astype(np.uint8)

    # run full-volume inference
    with torch.no_grad():
        inp = torch.from_numpy(vol_i[None,None]).to(device)
        out = model(inp)
        pred = (torch.sigmoid(out)>0.5).cpu().numpy()[0,0]

    z = vol_i.shape[2]//2
    plt.figure(figsize=(12,4))
    for i,(arr,ttl) in enumerate([
        (vol_i[:,:,z],   "T2"),
        (vol_m[:,:,z],   "GT Myoma"),
        (pred[:,:,z],    "Pred Myoma"),
    ]):
        ax = plt.subplot(1,3,i+1)
        ax.imshow(arr.T, cmap="gray", origin="lower")
        ax.set_title(ttl)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(args.save_path, exist_ok=True)
    plt.savefig(os.path.join(args.save_path, "final_example.png"))
    plt.show()

    # save history
    np.savez(os.path.join(args.save_path,"history.npz"), **history)

if __name__=="__main__":
    main()
