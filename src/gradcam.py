# ============================================================================
# OWNER: VISHNU - Proposal Step 4 / Goal 5 (Grad-CAM + share on object)
# Step 4 success: maps pass the randomization sanity check [14] (run with --sanity).
# Uses the masks from make_masks.py / make_masks_sam.py.
# ============================================================================
"""
Grad-CAM maps and how much of each map falls on the object.

    python src/gradcam.py checkpoints/resnet18_none_s0.pt              # share on object + example figure
    python src/gradcam.py checkpoints/resnet18_none_s0.pt --sanity     # also the randomization check [14]
    python src/gradcam.py --all

Saves results/gradcam/<run>.json and results/gradcam/<run>_examples.png

"share_on_object" = sum of the Grad-CAM map inside the object mask / sum of the whole map.
If the map were spread evenly, the share would equal the object's area, so we also report
"object_area" and "share_minus_area" (above 0 = the model looks at the object more than chance).

Sanity check (Adebayo et al.): if we replace the trained weights with random ones, the maps
should change a lot. We report the Spearman rank correlation between trained and random maps;
a low value (we use < 0.5) means the maps depend on what the model learned, so we can trust them.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import CHECKPOINT_DIR, CLASSES, IMG_SIZE, RESULTS_DIR
from data import final_transform
from evaluate import load_model
from make_masks import load_mask
from models import build_model, gradcam_layer
from utils import get_device, image_path, load_rgb, load_split, save_json


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """x: (1,3,H,W). Returns (map HxW in [0,1], predicted class)."""
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        pred = int(out.argmax(1))
        cls = pred if target is None else target
        out[0, cls].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # one weight per channel
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return (cam / cam.max() if cam.max() > 0 else cam), pred


def share_on_object(cam, mask):
    total = cam.sum()
    return float((cam * mask).sum() / total) if total > 0 else float("nan")


def analyze(ckpt_path, n_images=200, sanity=False, seed=0):
    device = get_device()
    model, ck = load_model(ckpt_path, device)
    run = ckpt_path.stem
    cam_fn = GradCAM(model, gradcam_layer(model, ck["model"]))
    tf = final_transform()

    test = load_split()["test"]
    random.Random(seed).shuffle(test)
    test = test[:n_images]

    if sanity:
        torch.manual_seed(seed)
        rand_model = build_model(ck["model"], pretrained=False).to(device)   # same network, random weights
        rand_cam = GradCAM(rand_model, gradcam_layer(rand_model, ck["model"]))

    shares, areas, correct, corrs, examples = [], [], [], [], []
    from scipy.stats import spearmanr
    for rel, label in test:
        img = load_rgb(image_path(rel))
        x = tf(img).unsqueeze(0).to(device)
        cam, pred = cam_fn(x)
        mask = load_mask(rel, (IMG_SIZE, IMG_SIZE)).astype(float)
        shares.append(share_on_object(cam, mask))
        areas.append(mask.mean())
        correct.append(pred == label)
        if sanity:
            rcam, _ = rand_cam(x, target=pred)
            corrs.append(spearmanr(cam.ravel(), rcam.ravel()).correlation)
        if len(examples) < 12:
            examples.append((img, cam, rel, CLASSES[pred]))

    shares, areas, correct = np.array(shares), np.array(areas), np.array(correct)
    out = {"run": run, "model": ck["model"], "aug": ck["aug"], "seed": ck["seed"], "n": len(test),
           "share_on_object": float(np.nanmean(shares)),
           "object_area": float(areas.mean()),
           "share_minus_area": float(np.nanmean(shares - areas)),
           "share_on_object_correct": float(np.nanmean(shares[correct])) if correct.any() else None,
           "share_on_object_wrong": float(np.nanmean(shares[~correct])) if (~correct).any() else None}
    if sanity:
        c = float(np.nanmean(np.abs(corrs)))
        out["sanity_rank_corr"] = c
        out["sanity_passed"] = c < 0.5
        print(f"Sanity check: |Spearman| trained vs random = {c:.3f} -> {'PASS' if c < 0.5 else 'FAIL'}")
    save_json(out, RESULTS_DIR / "gradcam" / f"{run}.json")
    save_examples(examples, RESULTS_DIR / "gradcam" / f"{run}_examples.png", run)
    print(f"{run}: share on object {out['share_on_object']:.3f} (object area {out['object_area']:.3f})")
    return out


def save_examples(examples, path, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for ax, (img, cam, rel, pred) in zip(axes.flat, examples):
        ax.imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{rel.split('/')[1]}\npred: {pred}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Grad-CAM: {title}")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    plt.close()


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=200, help="number of test images to use")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    paths = sorted(CHECKPOINT_DIR.glob("*_s*.pt")) if args.all else [Path(p) for p in args.checkpoints]
    if not paths:
        raise SystemExit("Give a checkpoint path or --all")
    for p in paths:
        analyze(p, n_images=args.n, sanity=args.sanity)


if __name__ == "__main__":
    main()
