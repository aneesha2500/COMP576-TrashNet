# ============================================================================
# OWNER: VISHNU - Proposal Step 4 / Goal 5 (Grad-CAM + share on object)
# Step 4 success: maps pass the randomization sanity check [14] (run with --sanity).
# Uses the masks from make_masks.py / make_masks_sam.py.
# ============================================================================
"""
Grad-CAM maps and how much of each map falls on the object.

    python src/gradcam.py checkpoints/resnet18_none_s0.pt              # share on object + example figure
    python src/gradcam.py checkpoints/resnet18_none_s0.pt --sanity     # also the randomization check [14]
    python src/gradcam.py --all --sanity --mask-dir data/masks_sam_test
    python src/gradcam.py --all --testset background --mask-dir data/masks_sam_test
    python src/gradcam.py --compare checkpoints/resnet18_none_s0.pt checkpoints/resnet18_background_s0.pt \
                                    checkpoints/resnet18_combined_s0.pt --mask-dir data/masks_sam_test

Saves results/gradcam/<run>.json and results/gradcam/<run>_examples.png for the clean test set,
and results/gradcam/<testset>/<run>.json (+ examples) for a modified test set.

--testset: "clean" (default) or a folder in data/test_shifts/ where the object does not move
(background, brightness_*, contrast_*, color_*). Rotations are not allowed because the mask would move.
--mask-dir: masks to use for the object (default: the same masks as training, from make_masks.load_mask).
We use data/masks_sam_test (SAM on every test image) because thresholding often marks white paper,
clear plastic and shadows wrongly even when the mask is not flagged as bad.

"share_on_object" = sum of the Grad-CAM map inside the object mask / sum of the whole map.
If the map were spread evenly, the share would equal the object's area, so we also report
"object_area" and "share_minus_area" (above 0 = the model looks at the object more than chance).

Sanity check (Adebayo et al.): if we replace the trained weights with random ones, the maps
should change a lot. We report the Spearman rank correlation between trained and random maps;
a low value (we use < 0.5) means the maps depend on what the model learned, so we can trust them.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CHECKPOINT_DIR, CLASSES, IMG_SIZE, RESULTS_DIR, ROOT, SHIFT_DIR
from data import final_transform
from evaluate import load_model
from make_masks import load_mask
from models import build_model
from utils import get_device, image_path, load_rgb, load_split, save_json


def cam_layer(model, name):
    """Layer whose output Grad-CAM uses: the last convolutional feature map.

    Small CNN: the last ReLU (features[-2]). Hooking the BatchNorm before it (models.gradcam_layer)
    gives the wrong gradients, because that ReLU works in place. ResNet-18: the last residual block.
    """
    return model.features[-2] if name == "smallcnn" else model.layer4[-1]


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


def get_mask(rel, mask_dir=None):
    """Object mask at IMG_SIZE x IMG_SIZE (True = object)."""
    if mask_dir is not None:
        p = Path(mask_dir) / rel.replace(".jpg", ".png")
        if p.exists():
            return np.asarray(Image.open(p).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
    return load_mask(rel, (IMG_SIZE, IMG_SIZE))


def test_image(rel, testset):
    if testset == "clean":
        return load_rgb(image_path(rel))
    return load_rgb(SHIFT_DIR / testset / rel)


def check_testset(testset):
    if testset == "clean":
        return
    if testset.startswith("rotate"):
        raise SystemExit("Rotated test sets are not supported: the object mask would need to rotate too.")
    if not (SHIFT_DIR / testset).exists():
        raise SystemExit(f"{SHIFT_DIR / testset} not found. Run: python src/make_test_shifts.py")


def out_dir_for(testset):
    return RESULTS_DIR / "gradcam" if testset == "clean" else RESULTS_DIR / "gradcam" / testset


def analyze(ckpt_path, n_images=None, sanity=False, seed=0, testset="clean", mask_dir=None):
    device = get_device()
    model, ck = load_model(ckpt_path, device)
    run = ckpt_path.stem
    cam_fn = GradCAM(model, cam_layer(model, ck["model"]))
    tf = final_transform()

    test = load_split()["test"]
    random.Random(seed).shuffle(test)
    test = test[:n_images] if n_images else test

    if sanity:
        torch.manual_seed(seed)
        rand_model = build_model(ck["model"], pretrained=False).to(device)   # same network, random weights
        rand_cam = GradCAM(rand_model, cam_layer(rand_model, ck["model"]))

    shares, areas, correct, corrs, examples = [], [], [], [], []
    per_class = {c: [] for c in CLASSES}
    from scipy.stats import spearmanr
    for rel, label in test:
        img = test_image(rel, testset)
        x = tf(img).unsqueeze(0).to(device)
        cam, pred = cam_fn(x)
        mask = get_mask(rel, mask_dir).astype(float)
        s = share_on_object(cam, mask)
        shares.append(s)
        areas.append(mask.mean())
        correct.append(pred == label)
        per_class[CLASSES[label]].append(s - mask.mean())
        if sanity:
            rcam, _ = rand_cam(x, target=pred)
            corrs.append(spearmanr(cam.ravel(), rcam.ravel()).correlation)
        if len(examples) < 12:
            examples.append((img, cam, rel, CLASSES[pred]))

    shares, areas, correct = np.array(shares), np.array(areas), np.array(correct)
    out = {"run": run, "model": ck["model"], "aug": ck["aug"], "seed": ck["seed"], "n": len(test),
           "testset": testset,
           "mask_dir": str(Path(mask_dir).relative_to(ROOT)) if mask_dir and Path(mask_dir).is_relative_to(ROOT)
           else (str(mask_dir) if mask_dir else "load_mask (training masks)"),
           "accuracy_on_these_images": float(correct.mean()),
           "share_on_object": float(np.nanmean(shares)),
           "object_area": float(areas.mean()),
           "share_minus_area": float(np.nanmean(shares - areas)),
           "share_on_object_correct": float(np.nanmean(shares[correct])) if correct.any() else None,
           "share_on_object_wrong": float(np.nanmean(shares[~correct])) if (~correct).any() else None,
           "share_minus_area_by_class": {c: (float(np.nanmean(v)) if v else None) for c, v in per_class.items()}}
    if sanity:
        c = float(np.nanmean(np.abs(corrs)))
        out["sanity_rank_corr"] = c
        out["sanity_rank_corr_std"] = float(np.nanstd(np.abs(corrs)))
        out["sanity_passed"] = c < 0.5
        print(f"Sanity check: |Spearman| trained vs random = {c:.3f} -> {'PASS' if c < 0.5 else 'FAIL'}")
    d = out_dir_for(testset)
    save_json(out, d / f"{run}.json")
    save_examples(examples, d / f"{run}_examples.png", f"{run} ({testset} test images)")
    print(f"{run} [{testset}]: share on object {out['share_on_object']:.3f} "
          f"(object area {out['object_area']:.3f})")
    return out


def save_examples(examples, path, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, cam, rel, pred) in zip(axes.flat, examples):
        ax.imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{rel.split('/')[1]}\npred: {pred}", fontsize=8)
    plt.suptitle(f"Grad-CAM: {title}")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    plt.close()


def compare_figure(ckpt_paths, mask_dir=None, testsets=("clean", "background"), n_images=3, seed=1):
    """Poster figure: the same test images, one row per model, clean vs. new background."""
    import matplotlib.pyplot as plt
    device = get_device()
    tf = final_transform()
    test = load_split()["test"]
    rng = random.Random(seed)
    by_class = {}
    for rel, label in test:
        by_class.setdefault(label, []).append(rel)
    picks = [rng.choice(by_class[c]) for c in rng.sample(sorted(by_class), n_images)]
    for ts in testsets:
        check_testset(ts)

    cols = [(rel, ts) for rel in picks for ts in testsets]
    fig, axes = plt.subplots(len(ckpt_paths), len(cols), figsize=(2.3 * len(cols), 2.5 * len(ckpt_paths)),
                             squeeze=False)
    for i, p in enumerate(ckpt_paths):
        model, ck = load_model(Path(p), device)
        cam_fn = GradCAM(model, cam_layer(model, ck["model"]))
        for j, (rel, ts) in enumerate(cols):
            img = test_image(rel, ts)
            cam, pred = cam_fn(tf(img).unsqueeze(0).to(device))
            ax = axes[i][j]
            ax.imshow(img.resize((IMG_SIZE, IMG_SIZE)))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.contour(get_mask(rel, mask_dir), levels=[0.5], colors="white", linewidths=0.8)
            true = rel.split("/")[0]
            ax.set_title(f"{ts}\ntrue {true}, pred {CLASSES[pred]}", fontsize=7,
                         color="black" if CLASSES[pred] == true else "red")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{ck['model']}\n{ck['aug']}", fontsize=9)
    plt.suptitle("Grad-CAM on the same test images (white line = object outline)", fontsize=10)
    plt.tight_layout()
    out = RESULTS_DIR / "gradcam" / "compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--all", action="store_true", help="every checkpoint in checkpoints/")
    ap.add_argument("--n", type=int, default=None, help="number of test images (default: all 379)")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--testset", default="clean")
    ap.add_argument("--mask-dir", default=None, help="e.g. data/masks_sam_test")
    ap.add_argument("--compare", action="store_true", help="make results/gradcam/compare.png from the given checkpoints")
    args = ap.parse_args()

    mask_dir = None
    if args.mask_dir:
        mask_dir = Path(args.mask_dir) if Path(args.mask_dir).is_absolute() else ROOT / args.mask_dir
        if not mask_dir.exists():
            raise SystemExit(f"{mask_dir} not found. Run make_masks_sam.py --split test --out-dir {args.mask_dir}")
    if args.compare:
        if not args.checkpoints:
            raise SystemExit("Give the checkpoints to compare (one row each).")
        compare_figure(args.checkpoints, mask_dir)
        return
    check_testset(args.testset)
    paths = sorted(CHECKPOINT_DIR.glob("*_s*.pt")) if args.all else [Path(p) for p in args.checkpoints]
    if not paths:
        raise SystemExit("Give a checkpoint path or --all")
    for p in paths:
        analyze(p, n_images=args.n, sanity=args.sanity, testset=args.testset, mask_dir=mask_dir)


if __name__ == "__main__":
    main()
