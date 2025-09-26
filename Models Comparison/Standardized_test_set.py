import argparse
import os
import random
import shutil
from pathlib import Path

def find_mask(image_path: Path, masks_dir: Path, suffix: str, exts: list):
    """
    Given an image Path, try to find the corresponding mask in masks_dir.
    First, try stem + suffix (e.g. '5_seg.png'). Then, try same stem,
    but never match the image file itself.
    """
    stem = image_path.stem

    # 1) Try with suffix (e.g. 5_seg.png)
    for ext in exts:
        candidate = masks_dir / f"{stem}{suffix}{ext}"
        if candidate.exists():
            return candidate

    # 2) Fall back to same stem—but avoid returning the image itself
    for ext in exts:
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists() and candidate.resolve() != image_path.resolve():
            return candidate

    return None


def main(images_dir, masks_dir, output_dir, test_ratio, suffix, exts, seed):
    random.seed(seed)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Gather image-mask pairs
    pairs = []
    for img_path in images_dir.iterdir():
        # Skip non-image files
        if img_path.suffix.lower() not in exts:
            continue
        # If images and masks are in the same folder, skip any mask files as "images"
        if images_dir.resolve() == masks_dir.resolve() and img_path.stem.endswith(suffix):
            continue

        mask_path = find_mask(img_path, masks_dir, suffix, exts)
        if mask_path is None:
            print(f"Warning: No mask found for {img_path.name}")
        else:
            pairs.append((img_path, mask_path))

    if not pairs:
        print("No image-mask pairs found. Exiting.")
        return

    # Determine number of test samples
    num_test = max(1, int(len(pairs) * test_ratio))
    test_samples = random.sample(pairs, num_test)

    # Prepare output directories
    test_img_out = output_dir / 'images'
    test_mask_out = output_dir / 'masks'
    test_img_out.mkdir(parents=True, exist_ok=True)
    test_mask_out.mkdir(parents=True, exist_ok=True)

    # Copy test files
    for img_path, mask_path in test_samples:
        shutil.copy(img_path, test_img_out / img_path.name)
        shutil.copy(mask_path, test_mask_out / mask_path.name)

    print(f"Saved {len(test_samples)} image-mask pairs to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split dataset into test set for image segmentation tasks."
    )
    parser.add_argument(
        '--images_dir', type=str, required=True,
        help='Path to the images directory'
    )
    parser.add_argument(
        '--masks_dir', type=str, required=True,
        help='Path to the masks directory'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path where the test set will be saved'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help='Proportion of data to use as test set (e.g., 0.1 for 10%)'
    )
    parser.add_argument(
        '--suffix', type=str, default='_seg',
        help='Suffix for mask filenames if different from images'
    )
    parser.add_argument(
        '--exts', nargs='+', default=['.jpg', '.png', '.tif'],
        help='Allowed image/mask file extensions'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    main(
        args.images_dir,
        args.masks_dir,
        args.output_dir,
        args.test_ratio,
        args.suffix,
        [e.lower() for e in args.exts],
        args.seed
    )
