# unet_segmentation/main.py

import argparse
import os
from train import run_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images',    type=str, required=True, help="Path to images folder")
    p.add_argument('--masks',     type=str, required=True, help="Path to masks folder")
    p.add_argument('--save_dir',  type=str, default='./checkpoints', help="Where to save outputs")
    p.add_argument('--epochs',    type=int, default=20)
    p.add_argument('--batch_size',type=int, default=8)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--img_size',  type=int, nargs=2, default=(256,256))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_training(
        image_dir=args.images,
        mask_dir =args.masks,
        save_dir =args.save_dir,
        epochs   =args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=tuple(args.img_size),
        device='cuda'
    )
