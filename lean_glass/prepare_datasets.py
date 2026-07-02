#!/usr/bin/env python3
"""Convert the Concrete Crack and PillQC datasets into the MVTec/GLASS layout.

Target layout (per category):
    <out>/<category>/train/good/*.png
    <out>/<category>/test/good/*.png
    <out>/<category>/test/<defect>/*.png
    <out>/<category>/ground_truth/<defect>/*_mask.png   (zero placeholders)

Neither source dataset provides pixel-level ground truth, so zero placeholder
masks are written only to satisfy the loader; pixel-level metrics are not
meaningful and should not be reported for these datasets. Image-level labels
are derived purely from the folder name (good vs defect), so image-level
AUROC/AP are valid.

Splits (fixed seed for reproducibility):
  Concrete Crack (Ozgenel): 'Negative' = normal, 'Positive' = crack.
      train/good = N_TRAIN random normal images; remaining normal -> test/good;
      CRACK_TEST random crack images -> test/crack.
  PillQC: images/{normal,dirt,chip}. normal -> train/test; dirt+chip -> test
      as separate defect folders (one model, defects pooled at scoring time).

Usage:
    python prepare_datasets.py --concrete /path/to/concrete --pillqc /path/to/pillQC --out /path/to/out
"""
import argparse
import os
import glob
import random
import shutil
import numpy as np
from PIL import Image

SEED = 0
N_TRAIN_CONCRETE = 1000      # normal images used for training (cf. MVTec scale)
N_TEST_GOOD_CONCRETE = 400   # held-out normal images for test
N_TEST_CRACK = 400           # crack images for test
PILL_TRAIN_FRAC = 0.6        # fraction of normal pills used for training
PILL_MIN_TEST_GOOD = 10      # never leave test/good empty; hold out at least this many


def ensure(d):
    os.makedirs(d, exist_ok=True)


def write_zero_mask(ref_image_path, mask_path):
    """Zero placeholder mask matching the image size."""
    with Image.open(ref_image_path) as im:
        w, h = im.size
    Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(mask_path)


def copy_set(paths, dst_dir):
    ensure(dst_dir)
    out = []
    for i, p in enumerate(paths):
        name = f"{i:04d}.png"
        with Image.open(p) as im:
            im.convert("RGB").save(os.path.join(dst_dir, name))
        out.append(os.path.join(dst_dir, name))
    return out


def make_masks(image_dir, mask_dir):
    ensure(mask_dir)
    for p in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        base = os.path.splitext(os.path.basename(p))[0]
        write_zero_mask(p, os.path.join(mask_dir, f"{base}_mask.png"))


def find_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, e))
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return sorted(set(files))


def prepare_concrete(src, out):
    rng = random.Random(SEED)
    # Ozgenel layout: Negative/ (normal), Positive/ (crack). Be tolerant of case.
    def cls_dir(*names):
        for n in names:
            for cand in (os.path.join(src, n), os.path.join(src, n.lower()), os.path.join(src, n.upper())):
                if os.path.isdir(cand):
                    return cand
        return None
    neg = cls_dir("Negative"); pos = cls_dir("Positive")
    assert neg and pos, f"Could not find Negative/Positive folders under {src}"
    normal = find_images(neg); crack = find_images(pos)
    rng.shuffle(normal); rng.shuffle(crack)
    cat = os.path.join(out, "concrete")
    # cap training count so a non-empty test/good is always retained
    n_train = min(N_TRAIN_CONCRETE, max(1, len(normal) - N_TEST_GOOD_CONCRETE))
    train_good = normal[:n_train]
    test_good = normal[n_train:n_train + N_TEST_GOOD_CONCRETE]
    test_crack = crack[:N_TEST_CRACK]
    if len(test_good) == 0:
        print("ERROR: concrete test/good is empty; image-level AUROC will be undefined (nan).")
    copy_set(train_good, os.path.join(cat, "train", "good"))
    copy_set(test_good, os.path.join(cat, "test", "good"))
    copy_set(test_crack, os.path.join(cat, "test", "crack"))
    make_masks(os.path.join(cat, "test", "crack"), os.path.join(cat, "ground_truth", "crack"))
    print(f"concrete: train/good={len(train_good)} test/good={len(test_good)} test/crack={len(test_crack)}")


def prepare_pill(src, out):
    rng = random.Random(SEED)
    img_root = os.path.join(src, "images")
    base = img_root if os.path.isdir(img_root) else src
    normal = find_images(os.path.join(base, "normal"))
    dirt = find_images(os.path.join(base, "dirt"))
    chip = find_images(os.path.join(base, "chip"))
    assert normal, f"Could not find normal pills under {base}/normal"
    rng.shuffle(normal)
    cat = os.path.join(out, "pill")
    n = len(normal)
    if n < 2 * PILL_MIN_TEST_GOOD:
        print(f"WARNING: only {n} normal pill images found under {base}/normal. "
              f"This is likely too few to train GLASS, and may indicate the path points "
              f"at 'sourceImages' rather than 'images/normal'. Proceeding anyway.")
    # proportional split with a guaranteed, non-empty test/good holdout
    n_test_good = max(PILL_MIN_TEST_GOOD, int(round(n * (1.0 - PILL_TRAIN_FRAC))))
    n_test_good = min(n_test_good, n - 1)            # keep at least 1 training image
    n_train = n - n_test_good
    train_good = normal[:n_train]
    test_good = normal[n_train:]
    copy_set(train_good, os.path.join(cat, "train", "good"))
    copy_set(test_good, os.path.join(cat, "test", "good"))
    for name, paths in (("dirt", dirt), ("chip", chip)):
        if paths:
            copy_set(paths, os.path.join(cat, "test", name))
            make_masks(os.path.join(cat, "test", name), os.path.join(cat, "ground_truth", name))
    print(f"pill: train/good={len(train_good)} test/good={len(test_good)} dirt={len(dirt)} chip={len(chip)}")
    if len(test_good) == 0:
        print("ERROR: test/good is empty; image-level AUROC will be undefined (nan).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concrete", help="root of Concrete Crack dataset (contains Negative/ Positive/)")
    ap.add_argument("--pillqc", help="root of pillQC repo (contains images/)")
    ap.add_argument("--out", required=True, help="output root in MVTec/GLASS layout")
    args = ap.parse_args()
    if args.concrete:
        prepare_concrete(args.concrete, args.out)
    if args.pillqc:
        prepare_pill(args.pillqc, args.out)
    print("done. Point GLASS --datapath at:", args.out)


if __name__ == "__main__":
    main()
