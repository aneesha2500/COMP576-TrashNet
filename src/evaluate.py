# ============================================================================
# OWNER: SÜLEYMAN - accuracy and accuracy drop on every test set
# Uses Aneesha's data/test_shifts/ and data/phone_photos/.
# ============================================================================
"""
Evaluate trained models on the clean test set, every modified test set, and our phone photos.

    python src/evaluate.py checkpoints/resnet18_none_s0.pt
    python src/evaluate.py --all          # every checkpoint in checkpoints/

Saves results/eval/<run>.json with, for each test set:
    acc, per-class acc, confusion matrix, and drop = clean acc - this acc  (our robustness measure)
"""
import argparse

import numpy as np
import torch

from config import CHECKPOINT_DIR, CLASSES, PHONE_DIR, RESULTS_DIR, SHIFT_DIR
from data import FolderSet, TrashNetSplit, make_loader
from models import build_model
from utils import get_device, load_split, save_json


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def scores(y, p):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    per_class = {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(CLASSES)}
    return {"n": int(len(y)), "acc": float((y == p).mean()) if len(y) else None,
            "per_class": per_class, "confusion": cm.tolist()}


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["model"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


def test_sets():
    """name -> dataset. Clean test first, then each modified set, then phone photos."""
    test_rels = {rel for rel, _ in load_split()["test"]}
    sets = {"clean": TrashNetSplit("test")}
    if SHIFT_DIR.exists():
        for d in sorted(p for p in SHIFT_DIR.iterdir() if p.is_dir()):
            sets[d.name] = FolderSet(d, only=test_rels)
    phone = FolderSet(PHONE_DIR) if PHONE_DIR.exists() else None
    if phone is not None and len(phone):
        sets["phone"] = phone
    return sets


def evaluate_checkpoint(ckpt_path, workers=2, batch_size=64):
    device = get_device()
    model, ck = load_model(ckpt_path, device)
    run = ckpt_path.stem if hasattr(ckpt_path, "stem") else str(ckpt_path)
    out = {"run": run, "model": ck["model"], "aug": ck["aug"], "seed": ck["seed"], "sets": {}}
    for name, ds in test_sets().items():
        y, p = predict(model, make_loader(ds, batch_size, workers=workers), device)
        out["sets"][name] = scores(y, p)
    clean = out["sets"]["clean"]["acc"]
    for name, s in out["sets"].items():
        s["drop"] = None if s["acc"] is None else clean - s["acc"]
    save_json(out, RESULTS_DIR / "eval" / f"{run}.json")
    print(f"{run}: " + ", ".join(f"{k}={v['acc']:.3f}" for k, v in out["sets"].items()))
    return out


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    paths = sorted(CHECKPOINT_DIR.glob("*_s*.pt")) if args.all else [Path(p) for p in args.checkpoints]
    if not paths:
        raise SystemExit("Give a checkpoint path or --all")
    for p in paths:
        evaluate_checkpoint(p, workers=args.workers)


if __name__ == "__main__":
    main()
