# ============================================================================
# OWNER: VISHNU - background segmentation with SAM (Feasibility: 'Separating objects')
# Fix the masks thresholding gets wrong. Run on Colab with a GPU.
# When done, re-run the visual check and tell Aneesha so she rebuilds the test sets
# (python src/make_test_shifts.py) with the better masks.
# Fixed masks go to data/masks_fixed/ -> COMMIT that folder so everyone gets them.
# ============================================================================
"""
Optional (Vishnu): redo the masks that thresholding got wrong using the Segment Anything Model.

    pip install git+https://github.com/facebookresearch/segment-anything.git
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
    python src/make_masks_sam.py                     # redo images in data/masks/bad_masks.txt
    python src/make_masks_sam.py --all               # redo every image (slow without a GPU)
    python src/make_masks_sam.py --files glass/glass3.jpg plastic/plastic10.jpg

Prompt: one box covering the middle of the image plus one point at the center,
since TrashNet objects are roughly centered. Keeps SAM's highest-scoring mask.
"""
import argparse

import numpy as np
from PIL import Image

from config import CHECKPOINT_DIR, CLASSES, MASK_DIR, MASK_FIX_DIR, TRASHNET_DIR
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"))
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    if args.files:
        rels = args.files
    elif args.all:
        rels = [f"{c}/{p.name}" for c in CLASSES for p in sorted((TRASHNET_DIR / c).glob("*.jpg"))]
    else:
        txt = (MASK_DIR / "bad_masks.txt").read_text().strip()
        rels = txt.split("\n") if txt else []
    print(f"Running SAM on {len(rels)} images")

    device = get_device()
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device if device.type == "cuda" else "cpu")
    predictor = SamPredictor(sam)

    for i, rel in enumerate(rels):
        img = np.asarray(Image.open(TRASHNET_DIR / rel).convert("RGB"))
        h, w = img.shape[:2]
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, h // 2]]), point_labels=np.array([1]),
            box=np.array([w * 0.1, h * 0.05, w * 0.9, h * 0.95]), multimask_output=True)
        m = masks[int(np.argmax(scores))]
        out = MASK_FIX_DIR / rel.replace(".jpg", ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((m * 255).astype(np.uint8)).save(out)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rels)}")
    print(f"Done. Saved to {MASK_FIX_DIR}. Look at a few, then commit data/masks_fixed/ to git.")


if __name__ == "__main__":
    main()
