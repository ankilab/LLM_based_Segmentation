# unet_segmentation/main.py
"""
Entry point.  Adapt DATA_ROOT and SAVE_DIR before running!
"""

from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm.auto import tqdm  # just to ensure tqdm is imported early
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import train_validate, test_phase


# ---------- configure here ----------------------------------------------------
#DATA_ROOT = Path("D:\qy44lyfe\LLM segmentation\Data sets\BAGLS\subset")   # folder containing 5000 *.png & *_seg.png
#DATA_ROOT = Path("D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images")
#DATA_ROOT = Path("D:\qy44lyfe\LLM segmentation\Data sets\Swallowing\images")
#DATA_ROOT = Path("D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_images\converted")
DATA_ROOT = Path("D:\qy44lyfe\LLM segmentation\Data sets\Skin cancer\subset\images")

#MASKS_DIR = DATA_ROOT
#MASKS_DIR = Path("D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks")
#MASKS_DIR = Path("D:\qy44lyfe\LLM segmentation\Data sets\Swallowing\masks")
#MASKS_DIR = Path("D:\qy44lyfe\LLM segmentation\Data sets\DRIVE\combined_masks\converted")
MASKS_DIR = Path("D:\qy44lyfe\LLM segmentation\Data sets\Skin cancer\subset\masks")
#MASK_SUFFIX = "_seg"                             # ""  → masks have *exact* same name
#MASK_SUFFIX = "_m"                         # "_m" → e.g. 0001_m.png
#MASK_SUFFIX = ""
MASK_SUFFIX = ""

#SAVE_DIR  = Path("D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\\BAGLS output")
#SAVE_DIR  = Path("D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\\Brain output")
#SAVE_DIR  = Path("D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\\Bolus output")
SAVE_DIR  = Path("D:\\qy44lyfe\\LLM segmentation\\Results\\2025\\GPT o3\\out of the box\\skin output")


RESIZE_TO     = (256, 256)
BATCH_SIZE    = 8
#EPOCHS        = 25
EPOCHS        = 3
LEARNING_RATE = 1e-3
RANDOM_SEED   = 42
# ------------------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def print_model_summary(model: torch.nn.Module,
#                         device: torch.device,
#                         img_size: tuple[int, int, int] = (1, 256, 256)) -> None:
#     """
#     Rudimentary summary: prints children names and the shape they output when
#     fed a single dummy image.  Works on CPU or GPU.
#     """
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(model)
#     print(f"\nTotal learnable parameters: {total_params:,}")
#
#     x = torch.randn(1, *img_size, device=device)  # <-- send dummy to *same* device
#     for name, layer in model.named_children():
#         x = layer(x)              # call layer, regardless of its type
#         print(f"{name:<12s} → {tuple(x.shape)}")

def main():
    set_seed(RANDOM_SEED)

    # 1. dataset ---------------------------------------------------------------
    full_dataset = SegmentationDataset(DATA_ROOT,
                                       masks_dir=MASKS_DIR,
                                       mask_suffix=MASK_SUFFIX,
                                       resize=RESIZE_TO)

    indices = list(range(len(full_dataset)))
    train_idx, tmp_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    val_idx, test_idx  = train_test_split(tmp_idx,   test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    print(
        f"Dataset sizes → train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)} "
        f"(total {len(full_dataset)})"
    )

    # 2. loaders ---------------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    for name, dl in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"{name.capitalize()} loader batches: {len(dl)}")

    # 3. model & optimiser -----------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_channels=1, out_channels=1, init_features=32).to(device)
    #print_model_summary(model, device)
    summary(model,
            input_size=(1, 1, 256, 256),
            col_names=("input_size", "output_size", "num_params"),
            device=device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. training --------------------------------------------------------------
    train_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=EPOCHS,
        save_dir=SAVE_DIR,
    )

    # 5. save model ------------------------------------------------------------
    torch.save(model, SAVE_DIR / "unet_full_model.pth")
    torch.save(model.state_dict(), SAVE_DIR / "unet_state_dict.pth")
    print(f"\nModel saved to {SAVE_DIR}")

    # 6. testing  --------------------------------------------------------------
    test_phase(model, test_loader, criterion, device, SAVE_DIR)


if __name__ == "__main__":
    main()
