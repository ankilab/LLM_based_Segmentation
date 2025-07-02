import os
import time
import argparse
import torch
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import UMDDataset
from model import UNet3D
from train import train_epoch, validate_epoch, test_epoch, plot_predictions

def gather_pairs(root):
    imgs = sorted(glob.glob(os.path.join(root, "imagesTr","*.nii.gz")))
    msks = sorted(glob.glob(os.path.join(root, "labelsTr","*.nii.gz")))
    # map base key -> mask
    mmap = {os.path.basename(m)[:-len("_t2.nii.gz")]: m for m in msks}
    pairs = []
    for im in imgs:
        key = os.path.basename(im)[:-len("_t2_0000.nii.gz")]
        if key in mmap:
            pairs.append((im, mmap[key]))
    return pairs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = gather_pairs(args.data_root)
    imgs, msks = zip(*pairs)

    # split
    idxs = list(range(len(imgs)))
    tr, rest = train_test_split(idxs, test_size=0.2, random_state=args.seed)
    vl, te  = train_test_split(rest, test_size=0.5, random_state=args.seed)

    def pick(lst, idxs): return [lst[i] for i in idxs]

    tr_ds = UMDDataset(pick(imgs,tr), pick(msks,tr))
    vl_ds = UMDDataset(pick(imgs,vl), pick(msks,vl))
    te_ds = UMDDataset(pick(imgs,te), pick(msks,te))

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    vl_ld = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=1,          shuffle=False, num_workers=2)

    model = UNet3D(1,1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss = train_epoch(model, tr_ld, opt, device)
        vl_loss, vl_dices = validate_epoch(model, vl_ld, device)
        print(f"Epoch {ep+1}/{args.epochs}  train_loss={tr_loss:.3f}  val_loss={vl_loss:.3f}  meanDice={np.mean(vl_dices):.3f}  time={(time.time()-t0):.1f}s")

    # final test
    test_dices = test_epoch(model, te_ld, device)
    print("Test mean Dice:", np.mean(test_dices))

    # plots
    #plot_predictions(model, te_ld, device, args.save_path, n_examples=args.n_cases)
    plot_predictions(model, te_ld, device, args.save_path)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--save_path",  type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n_cases",    type=int,   default=5)
    args = p.parse_args()
    main(args)
