# ============================================================================
# OWNER: VISHNU - background segmentation with SAM (Feasibility: 'Separating objects')
# Fix the masks thresholding gets wrong. Run on Colab with a GPU.
# When done, re-run the visual check and tell Aneesha so she rebuilds the test sets
# (python src/make_test_shifts.py) with the better masks.
# Fixed masks go to data/masks_fixed/ -> COMMIT that folder so everyone gets them.
# ============================================================================
"""
Redo object masks with the Segment Anything Model (SAM) [Kirillov et al., 2023].

    pip install git+https://github.com/facebookresearch/segment-anything.git
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P /content/sam/

    # 1) the 337 images thresholding got wrong (data/masks/bad_masks.txt) -> data/masks_fixed/
    python src/make_masks_sam.py --checkpoint /content/sam/sam_vit_h_4b8939.pth --model-type vit_h --check

    # 2) every TEST image -> data/masks_sam_test/ (only used by gradcam.py --mask-dir, not by training)
    python src/make_masks_sam.py --split test --out-dir data/masks_sam_test \
        --checkpoint /content/sam/sam_vit_h_4b8939.pth --model-type vit_h --check

    # other ways to pick images
    python src/make_masks_sam.py --all
    python src/make_masks_sam.py --files glass/glass3.jpg plastic/plastic10.jpg

    # after looking at the check pages: masks SAM still got wrong
    #   write them (e.g. glass/glass3.jpg), one per line, in <out-dir>/sam_rejected.txt, then
    python src/make_masks_sam.py --apply-rejects            # (add --out-dir for the test masks)

Prompt: one box over the middle of the image plus one point at the center, since TrashNet
objects are roughly centered. SAM returns 3 candidate masks; we keep the highest-scoring one
whose area is plausible (5-95% of the image, the same rule make_masks.py uses to flag bad masks).
If none is plausible we retry with the center point only. If that also fails the image is
marked "still_bad" and its threshold mask is kept (no file is written for it).

Outputs (per run, in results/sam/<out-dir name>/):
    stats.csv        one row per image: areas, SAM score, which attempt worked, IoU with the threshold mask
    summary.json     counts for the report (how many fixed, how many still bad, per class)
    check_XX.png     pages of 40 overlays (red = background) to look through by eye
    before_after.png 8 examples, threshold mask vs SAM mask (for the poster / report)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASSES, MASK_DIR, MASK_FIX_DIR, RESULTS_DIR, ROOT, TRASHNET_DIR
from utils import get_device, load_split

MIN_AREA, MAX_AREA = 0.05, 0.95   # same rule as make_masks.py


def plausible(m):
    return MIN_AREA < m.mean() < MAX_AREA


def segment(predictor, img):
    """img: HxWx3 uint8. Returns (mask bool HxW, score, attempt name)."""
    h, w = img.shape[:2]
    predictor.set_image(img)
    point = np.array([[w // 2, h // 2]])
    box = np.array([w * 0.1, h * 0.05, w * 0.9, h * 0.95])

    masks, scores, _ = predictor.predict(point_coords=point, point_labels=np.array([1]),
                                         box=box, multimask_output=True)
    best = int(np.argmax(scores))
    if plausible(masks[best]):
        return masks[best], float(scores[best]), "box+point"
    ok = [i for i in range(len(masks)) if plausible(masks[i])]
    if ok:
        i = max(ok, key=lambda k: scores[k])
        return masks[i], float(scores[i]), "box+point (2nd choice)"

    masks2, scores2, _ = predictor.predict(point_coords=point, point_labels=np.array([1]),
                                           multimask_output=True)
    ok = [i for i in range(len(masks2)) if plausible(masks2[i])]
    if ok:
        i = max(ok, key=lambda k: scores2[k])
        return masks2[i], float(scores2[i]), "point only"
    return masks[best], float(scores[best]), "still_bad"


def threshold_mask_of(rel):
    p = MASK_DIR / rel.replace(".jpg", ".png")
    return np.asarray(Image.open(p)) > 127 if p.exists() else None


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def pick_images(args):
    if args.files:
        return args.files
    if args.all:
        return [f"{c}/{p.name}" for c in CLASSES for p in sorted((TRASHNET_DIR / c).glob("*.jpg"))]
    if args.split:
        return [rel for rel, _ in load_split()[args.split]]
    bad = MASK_DIR / "bad_masks.txt"
    if not bad.exists():
        raise SystemExit(f"{bad} not found. Run: python src/make_masks.py")
    txt = bad.read_text().strip()
    return txt.split("\n") if txt else []


def overlay(img, m):
    a = img.astype(float).copy()
    a[~m] = a[~m] * 0.25 + np.array([255, 0, 0]) * 0.75     # background shown red
    return a.astype(np.uint8)


def save_check_pages(rows, out_dir, per_page=40):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("check_*.png"):
        old.unlink()
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for k, page in enumerate(pages, 1):
        fig, axes = plt.subplots(5, 8, figsize=(20, 9.5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, page):
            img = np.asarray(Image.open(TRASHNET_DIR / r["rel"]).convert("RGB"))
            ax.imshow(overlay(img, r["_mask"]))
            flag = "  STILL BAD" if r["attempt"] == "still_bad" else ""
            ax.set_title(f"#{r['idx']} {r['rel']}{flag}", fontsize=7, color="red" if flag else "black")
        plt.suptitle(f"SAM masks, page {k}/{len(pages)}. Red = background. Write the names of wrong ones "
                     f"in sam_rejected.txt", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / f"check_{k:02d}.png", dpi=60)
        plt.close()
    print(f"Saved {len(pages)} check pages to {out_dir}")


def save_before_after(rows, path, n=8):
    """Images where SAM changed the mask most (lowest IoU with the threshold mask)."""
    import matplotlib.pyplot as plt
    cand = [r for r in rows if r["iou_with_threshold"] == r["iou_with_threshold"] and r["attempt"] != "still_bad"]
    cand = sorted(cand, key=lambda r: r["iou_with_threshold"])
    picked, seen = [], set()
    for r in cand:                                   # one per class first, then fill up
        c = r["rel"].split("/")[0]
        if c not in seen:
            picked.append(r)
            seen.add(c)
    picked += [r for r in cand if r not in picked][: n - len(picked)]
    picked = picked[:n]
    if not picked:
        return
    fig, axes = plt.subplots(2, len(picked), figsize=(2.6 * len(picked), 4.4), squeeze=False)
    for j, r in enumerate(picked):
        img = np.asarray(Image.open(TRASHNET_DIR / r["rel"]).convert("RGB"))
        axes[0][j].imshow(overlay(img, r["_thr"]))
        axes[1][j].imshow(overlay(img, r["_mask"]))
        axes[0][j].set_title(r["rel"].split("/")[1], fontsize=8)
        for i in range(2):
            axes[i][j].axis("off")
    axes[0][0].text(-0.08, 0.5, "threshold", transform=axes[0][0].transAxes, rotation=90, va="center", ha="right")
    axes[1][0].text(-0.08, 0.5, "SAM", transform=axes[1][0].transAxes, rotation=90, va="center", ha="right")
    plt.suptitle("Object masks before (threshold) and after SAM. Red = background.")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"Saved {path}")


def summarize(rows, out_dir_name, model_type):
    def count(rs):
        fixed = sum(r["attempt"] != "still_bad" for r in rs)
        return {"n": len(rs), "fixed": fixed, "still_bad": len(rs) - fixed}
    s = {"masks_dir": out_dir_name, "sam_model": model_type, **count(rows),
         "by_attempt": {a: sum(r["attempt"] == a for r in rows)
                        for a in ["box+point", "box+point (2nd choice)", "point only", "still_bad"]},
         "by_class": {c: count([r for r in rows if r["rel"].startswith(c + "/")]) for c in CLASSES},
         "threshold_was_flagged": sum(r["threshold_flagged"] for r in rows)}
    ious = [r["iou_with_threshold"] for r in rows if r["iou_with_threshold"] == r["iou_with_threshold"]]
    if ious:
        s["iou_with_threshold_mean"] = float(np.mean(ious))
        s["iou_with_threshold_below_0.8"] = int(sum(i < 0.8 for i in ious))
    return s


def run_sam(args, out_dir, res_dir):
    from segment_anything import SamPredictor, sam_model_registry

    rels = pick_images(args)
    print(f"Running SAM ({args.model_type}) on {len(rels)} images -> {out_dir}")
    device = get_device()
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device if device.type == "cuda" else "cpu")
    predictor = SamPredictor(sam)

    rows = []
    for i, rel in enumerate(rels):
        img = np.asarray(Image.open(TRASHNET_DIR / rel).convert("RGB"))
        m, score, attempt = segment(predictor, img)
        thr = threshold_mask_of(rel)
        out = out_dir / rel.replace(".jpg", ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        if attempt != "still_bad":
            Image.fromarray((m * 255).astype(np.uint8)).save(out)
        elif out.exists():
            out.unlink()                       # keep the threshold mask for this image
        rows.append({"idx": i, "rel": rel, "attempt": attempt, "sam_score": round(score, 4),
                     "sam_area": round(float(m.mean()), 4),
                     "threshold_area": None if thr is None else round(float(thr.mean()), 4),
                     "threshold_flagged": bool(thr is not None and not plausible(thr)),
                     "iou_with_threshold": float("nan") if thr is None else round(iou(m, thr), 4),
                     "_mask": m, "_thr": thr if thr is not None else np.ones_like(m)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rels)}")

    res_dir.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if not k.startswith("_")] if rows else []
    with open(res_dir / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    s = summarize(rows, str(out_dir.relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir),
                  args.model_type)
    (res_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ["n", "fixed", "still_bad", "by_attempt"]}, indent=2))
    if args.check and rows:
        save_check_pages(rows, res_dir)
        save_before_after(rows, res_dir / "before_after.png")
    print(f"Done. Masks in {out_dir}, stats in {res_dir}. Look through the check pages next.")


def apply_rejects(out_dir, res_dir):
    """Remove the SAM masks you marked as wrong (their threshold mask is used again)."""
    rej_file = out_dir / "sam_rejected.txt"
    if not rej_file.exists():
        raise SystemExit(f"Write the wrong ones in {rej_file} first (one like glass/glass3.jpg per line).")
    rejected = [r.strip() for r in rej_file.read_text().splitlines() if r.strip() and not r.startswith("#")]
    keep = ROOT / "data" / f"{out_dir.name}_rejected"
    moved = 0
    for rel in rejected:
        p = out_dir / rel.replace(".jpg", ".png")
        if p.exists():
            (keep / p.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(keep / p.parent.name / p.name))
            moved += 1
    summ = res_dir / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        s["rejected_after_visual_check"] = rejected
        s["fixed_after_visual_check"] = s["fixed"] - moved
        summ.write_text(json.dumps(s, indent=2))
    print(f"{len(rejected)} listed, {moved} SAM masks moved to {keep} (their threshold masks are used again).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/content/sam/sam_vit_h_4b8939.pth")
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--all", action="store_true", help="every TrashNet image")
    ap.add_argument("--split", choices=["train", "val", "test"], help="every image in one part of the split")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out-dir", default=str(MASK_FIX_DIR),
                    help="where masks go (default data/masks_fixed, which training and test sets use)")
    ap.add_argument("--check", action="store_true", help="save check pages and before/after figure")
    ap.add_argument("--apply-rejects", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    res_dir = RESULTS_DIR / "sam" / out_dir.name
    if args.apply_rejects:
        apply_rejects(out_dir, res_dir)
    else:
        run_sam(args, out_dir, res_dir)


if __name__ == "__main__":
    main()
