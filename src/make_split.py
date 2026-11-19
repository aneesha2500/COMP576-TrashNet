# ============================================================================
# OWNER: ANEESHA - Proposal Step 1 (data preparation)
# Run ONCE, then commit splits/split.json so all of us use the same 70/15/15 split.
# Success check from the proposal: class proportions in each split match the full dataset
# (the script prints the counts per class).
# ============================================================================
"""
Make the fixed 70/15/15 split, stratified by class.

    python src/make_split.py

Writes splits/split.json. Run this ONCE, commit the file, and everyone uses the same split.
"""
import json
import random

from config import CLASSES, SPLIT_FILE, TRASHNET_DIR


def main(seed=0, train=0.70, val=0.15):
    if SPLIT_FILE.exists():
        print(f"{SPLIT_FILE} already exists. Delete it first if you really want a new split.")
        return
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for c in CLASSES:
        files = sorted(p.name for p in (TRASHNET_DIR / c).glob("*.jpg"))
        if not files:
            raise SystemExit(f"No images in {TRASHNET_DIR / c}. Run src/get_data.py first.")
        rng.shuffle(files)
        n_tr = round(len(files) * train)
        n_va = round(len(files) * val)
        split["train"] += [f"{c}/{f}" for f in files[:n_tr]]
        split["val"] += [f"{c}/{f}" for f in files[n_tr:n_tr + n_va]]
        split["test"] += [f"{c}/{f}" for f in files[n_tr + n_va:]]
        print(f"{c:10s} total={len(files):4d} train={n_tr:4d} val={n_va:3d} test={len(files) - n_tr - n_va:3d}")
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    print(f"Saved {SPLIT_FILE}")


if __name__ == "__main__":
    main()
