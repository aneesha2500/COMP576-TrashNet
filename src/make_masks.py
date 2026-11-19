# ============================================================================
# OWNER: ANEESHA (run it) + VISHNU (check and fix the masks) - Proposal Step 1
# Aneesha: run with --check and look at results/mask_check.png.
# Vishnu: images listed in data/masks/bad_masks.txt (white plastic, clear glass, foil)
#         get redone with SAM in make_masks_sam.py. You can also tune `light` and `gray`
#         in threshold_mask() if many masks are wrong.
# ============================================================================
"""
Find the object in every TrashNet image (needed for background replacement
and for the Grad-CAM "share on object" measure).

    python src/make_masks.py            # makes data/masks/<class>/<name>.png
    python src/make_masks.py --check    # also saves results/mask_check.png (50 random masks)

Method: TrashNet backgrounds are light and almost gray. We mark light, gray pixels that are
connected to the image border as background. Everything else is the object.
Masks that look wrong (object covers <5% or >95% of the image) are listed in
data/masks/bad_masks.txt. Those can be redone with SAM (src/make_masks_sam.py), which saves to data/masks_fixed/.
A mask in data/masks_fixed/ is always used instead of the thresholded one.
"""
import argparse
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import CLASSES, MASK_DIR, MASK_FIX_DIR, RESULTS_DIR, TRASHNET_DIR


def threshold_mask(img, light=140, gray=40, step=8):
    """Returns a bool array, True = object."""
    a = np.asarray(img).astype(np.int16)
    is_light = a.mean(axis=2) > light
    is_gray = (a.max(axis=2) - a.min(axis=2)) < gray
    cand = Image.fromarray(((is_light & is_gray) * 255).astype(np.uint8)).copy()
    w, h = cand.size
    border = ([(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] +
              [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)])
    for xy in border:
        if cand.getpixel(xy) == 255:
            ImageDraw.floodfill(cand, xy, 128)
    background = np.asarray(cand) == 128
    obj = Image.fromarray((~background * 255).astype(np.uint8))
    # smooth: remove specks and fill small holes
    obj = obj.filter(ImageFilter.MedianFilter(5))
    return np.asarray(obj) > 127


def mask_path(rel):
    return MASK_DIR / rel.replace(".jpg", ".png")


def load_mask(rel, size=None):
    """Load a saved mask as a bool array (True = object). Uses Vishnu's fixed mask if there is one."""
    fixed = MASK_FIX_DIR / rel.replace(".jpg", ".png")
    m = Image.open(fixed if fixed.exists() else mask_path(rel))
    if size is not None:
        m = m.resize(size, Image.NEAREST)
    return np.asarray(m) > 127


def save_check_grid(rels, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 10, figsize=(20, 8))
    for ax, rel in zip(axes.flat, rels):
        img = np.asarray(Image.open(TRASHNET_DIR / rel).convert("RGB")).astype(float)
        m = load_mask(rel)
        overlay = img.copy()
        overlay[~m] = overlay[~m] * 0.25 + np.array([255, 0, 0]) * 0.75   # background shown red
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(rel.split("/")[1], fontsize=7)
        ax.axis("off")
    plt.suptitle("Red = background. Check that the red never covers the object.")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=80)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="save a grid of 50 random masks")
    args = ap.parse_args()

    bad, all_rels = [], []
    for c in CLASSES:
        (MASK_DIR / c).mkdir(parents=True, exist_ok=True)
        for p in sorted((TRASHNET_DIR / c).glob("*.jpg")):
            rel = f"{c}/{p.name}"
            all_rels.append(rel)
            m = threshold_mask(Image.open(p).convert("RGB"))
            Image.fromarray((m * 255).astype(np.uint8)).save(mask_path(rel))
            fixed = (MASK_FIX_DIR / rel.replace(".jpg", ".png")).exists()
            if not fixed and not 0.05 < m.mean() < 0.95:
                bad.append(rel)
        print(f"{c}: done")
    (MASK_DIR / "bad_masks.txt").write_text("\n".join(bad))
    print(f"{len(all_rels)} masks saved, {len(bad)} look wrong (listed in {MASK_DIR / 'bad_masks.txt'})")

    if args.check:
        random.seed(0)
        save_check_grid(random.sample(all_rels, 50), RESULTS_DIR / "mask_check.png")


if __name__ == "__main__":
    main()
