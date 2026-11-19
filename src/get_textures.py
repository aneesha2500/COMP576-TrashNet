# ============================================================================
# OWNER: ANEESHA - Proposal Step 1 (backgrounds for the modified test sets)
# Keep train/ and test/ textures separate (design requirement: test backgrounds are
# never used in training). You can add your own photos of floors, desks, grass to
# data/textures/test/.
# ============================================================================
"""
Get background textures, split into two groups that never overlap:
    data/textures/train/  -> used only for the "background" training augmentation (Goal 3)
    data/textures/test/   -> used only to build the modified test set (Goal 2)

    python src/get_textures.py                 # downloads the DTD texture dataset (~600 MB)
    python src/get_textures.py --procedural    # no download: makes simple synthetic textures

DTD has 47 texture categories. Every other category goes to train, the rest to test,
so test backgrounds are a different KIND of surface, not just different photos.
You can also drop your own photos (floor, desk, grass...) into data/textures/test/.
"""
import argparse
import shutil

import numpy as np
from PIL import Image

from config import DATA_DIR, TEXTURE_DIR

SIZE = (512, 384)   # same size as TrashNet images


def from_dtd(per_category):
    from torchvision.datasets import DTD
    root = DATA_DIR / "dtd_download"
    ds = DTD(root=str(root), split="train", download=True)
    cats = sorted(set(ds.classes))
    by_cat = {c: [] for c in cats}
    for path, label in zip(ds._image_files, ds._labels):
        by_cat[ds.classes[label]].append(path)
    for i, c in enumerate(cats):
        group = "train" if i % 2 == 0 else "test"
        out = TEXTURE_DIR / group
        out.mkdir(parents=True, exist_ok=True)
        for j, p in enumerate(sorted(by_cat[c])[:per_category]):
            Image.open(p).convert("RGB").resize(SIZE).save(out / f"{c}_{j}.jpg", quality=90)
    shutil.rmtree(root, ignore_errors=True)


def procedural(n_per_group):
    """Simple textures made with math (stripes, noise, checks). Not as realistic as DTD."""
    rng = np.random.default_rng(0)
    h, w = SIZE[1], SIZE[0]
    y, x = np.mgrid[0:h, 0:w]

    def make(kind, base):
        if kind == "wood":
            g = 0.5 + 0.5 * np.sin(x / rng.uniform(5, 15) + 3 * np.sin(y / rng.uniform(20, 60)))
        elif kind == "noise":
            small = rng.random((h // 16 + 1, w // 16 + 1))
            g = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255
        elif kind == "checks":
            s = rng.integers(20, 60)
            g = ((x // s + y // s) % 2) * 0.6 + 0.2
        else:  # speckle, like concrete
            g = 0.5 + 0.25 * rng.standard_normal((h, w))
        g = np.clip(g, 0, 1)[..., None]
        return (base * (0.5 + 0.7 * g)).clip(0, 255).astype(np.uint8)

    groups = {"train": ["wood", "checks"], "test": ["noise", "speckle"]}
    for group, kinds in groups.items():
        out = TEXTURE_DIR / group
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_group):
            base = rng.uniform(40, 200, size=3)
            Image.fromarray(make(kinds[i % 2], base)).save(out / f"{kinds[i % 2]}_{i}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedural", action="store_true")
    ap.add_argument("--per-category", type=int, default=20)
    args = ap.parse_args()
    if args.procedural:
        procedural(60)
    else:
        from_dtd(args.per_category)
    for g in ["train", "test"]:
        print(f"{g}: {len(list((TEXTURE_DIR / g).glob('*.jpg')))} textures in {TEXTURE_DIR / g}")


if __name__ == "__main__":
    main()
