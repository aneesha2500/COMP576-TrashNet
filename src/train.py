# ============================================================================
# OWNER: SÜLEYMAN - Proposal Steps 2-3 (baselines + augmentation experiments)
# Step 2 success: ResNet-18 'none' gets accuracy close to published TrashNet results [15, 16].
# Step 3: run every model x augmentation x 3 seeds (scripts/run_all.sh).
# ============================================================================
"""
Train one model with one augmentation setting and one seed.

    python src/train.py --model resnet18 --aug none --seed 0
    python src/train.py --model smallcnn --aug combined --seed 2 --epochs 60

Saves:
    checkpoints/<model>_<aug>_s<seed>.pt      best model (highest validation accuracy)
    results/train/<model>_<aug>_s<seed>.json  loss/accuracy for every epoch
Then it runs src/evaluate.py on the best model unless you pass --no-eval.
Loss: cross-entropy. Optimizer: Adam. Early stopping on validation accuracy.
The rare 'trash' class is oversampled (turn off with --no-oversample).
"""
import argparse
import time

import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, RESULTS_DIR
from data import Augment, TrashNetSplit, make_loader, oversampling_sampler
from models import build_model
from utils import get_device, save_json, set_seed

DEFAULT_LR = {"smallcnn": 1e-3, "resnet18": 1e-4}
DEFAULT_EPOCHS = {"smallcnn": 60, "resnet18": 25}


def run_name(model, aug, seed):
    return f"{model}_{aug}_s{seed}"


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train(args):
    set_seed(args.seed)
    device = get_device()
    name = run_name(args.model, args.aug, args.seed)
    lr = args.lr or DEFAULT_LR[args.model]
    epochs = args.epochs or DEFAULT_EPOCHS[args.model]
    print(f"== {name} on {device} | lr={lr} epochs<={epochs}")

    train_ds = TrashNetSplit("train", augment=Augment(args.aug))
    val_ds = TrashNetSplit("val")
    sampler = None if args.no_oversample else oversampling_sampler(train_ds.items)
    train_dl = make_loader(train_ds, args.batch_size, shuffle=True, sampler=sampler, workers=args.workers)
    val_dl = make_loader(val_ds, args.batch_size, workers=args.workers)

    model = build_model(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"{name}.pt"
    partial = CHECKPOINT_DIR / f"{name}.partial"   # renamed to .pt only when training finishes
    history, best, bad_epochs = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = correct = seen = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        val_acc = accuracy(model, val_dl, device)
        sched.step(val_acc)
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "train_acc": correct / seen,
                        "val_acc": val_acc, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:3d} | loss {total_loss / seen:.3f} | train {correct / seen:.3f} "
              f"| val {val_acc:.3f} | {time.time() - t0:.0f}s")
        if val_acc > best:
            best, bad_epochs = val_acc, 0
            torch.save({"model": args.model, "aug": args.aug, "seed": args.seed,
                        "state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, partial)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop: no improvement for {args.patience} epochs")
                break

    import shutil
    shutil.move(str(partial), str(ckpt))   # only finished runs get a .pt file
    save_json({"run": name, "best_val_acc": best, "lr": lr, "args": vars(args), "epochs_run": len(history),
               "epochs": history}, RESULTS_DIR / "train" / f"{name}.json")
    print(f"Best val acc {best:.3f}. Saved {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["smallcnn", "resnet18"])
    ap.add_argument("--aug", default="none",
                    choices=["none", "standard", "photometric", "geometric", "background", "combined"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="early stopping patience (epochs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    ckpt = train(args)
    if not args.no_eval:
        from evaluate import evaluate_checkpoint
        evaluate_checkpoint(ckpt, workers=args.workers)


if __name__ == "__main__":
    main()
