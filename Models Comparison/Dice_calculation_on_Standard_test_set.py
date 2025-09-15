
import os
import glob
import argparse
import importlib.util

import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

'''
This code is to import each models model.py and model state dict, and run it on the standardized test sets
to calculate the dice scores for evaluation, and saving them in excel files
'''
def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred_bin   = (pred > 0.5).astype(np.float32)
    target_bin = (target > 0.5).astype(np.float32)
    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin)
    return (2 * intersection + smooth) / (union + smooth)

def load_model(model_py: str, weights_pth: str, device: torch.device):
    spec = importlib.util.spec_from_file_location("model_module", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UNet()
    state = torch.load(weights_pth, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

def find_pairs(img_dir: str, mask_dir: str, suffix: str) -> list:
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs = []
    for img_path in img_paths:
        base, ext = os.path.splitext(os.path.basename(img_path))
        m1 = os.path.join(mask_dir, base + ext)
        m2 = os.path.join(mask_dir, base + suffix + ext)
        if   os.path.exists(m1):
            mask_path = m1
        elif os.path.exists(m2):
            mask_path = m2
        else:
            raise FileNotFoundError(f"No mask for {base!r}; tried {m1!r} and {m2!r}")
        pairs.append((img_path, mask_path))
    return pairs

def load_gray_tensor(img_path: str, device: torch.device):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, h, w

def load_and_resize_mask(mask_path: str, target_h: int, target_w: int) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    if m.size != (target_w, target_h):
        m = m.resize((target_w, target_h), resample=Image.NEAREST)
    arr = np.array(m, dtype=np.float32) / 255.0
    return arr

def evaluate(model, pairs: list, device: torch.device) -> list:
    results = []
    for img_path, mask_path in tqdm(pairs, desc="Evaluating"):
        x, h_in, w_in = load_gray_tensor(img_path, device)
        with torch.no_grad():
            out = model(x)
            pred = torch.sigmoid(out).cpu().numpy()[0, 0]
        h_out, w_out = pred.shape
        mask_arr = load_and_resize_mask(mask_path, h_out, w_out)
        score = dice_score(pred, mask_arr)
        results.append((os.path.basename(img_path), float(score)))
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate one UNet model and save Dice scores")
    parser.add_argument("--model-py",     required=True, help="Path to model.py")
    parser.add_argument("--weights-pth",  required=True, help="Path to .pth state_dict")
    parser.add_argument("--test-images",  required=True, help="Dir of test images")
    parser.add_argument("--test-masks",   required=True, help="Dir of test masks")
    parser.add_argument("--mask-suffix",  default="_m", help="Suffix before mask extension")
    parser.add_argument("--output-excel", required=True, help="Output .xlsx file path")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    device = torch.device(args.device)

    model  = load_model(args.model_py, args.weights_pth, device)
    pairs  = find_pairs(args.test_images, args.test_masks, args.mask_suffix)
    scores = evaluate(model, pairs, device)

    # Save per-image scores only
    df = pd.DataFrame(scores, columns=["image", "dice"])
    df.to_excel(args.output_excel, index=False)
    print(f"Saved per-image Dice scores to {args.output_excel}")

    # Compute and print mean ± std
    mean = df["dice"].mean()
    std  = df["dice"].std(ddof=0)
    print(f"Mean Dice: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
