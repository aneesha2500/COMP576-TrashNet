# ============================================================================
# OWNER: SÜLEYMAN - Proposal Step 5 (tables, plots, mean ± std over seeds)
# Output (results/summary.md, results/plots/) goes on the poster and in the report.
# ============================================================================
"""
Combine all results into tables and plots for the poster and report.

    python src/analyze.py

Reads results/eval/*.json (and results/gradcam/*.json if present). Writes:
    results/summary_accuracy.csv   mean and std over seeds of accuracy on every test set
    results/summary_drop.csv       mean and std over seeds of the accuracy drop
    results/summary.md             the same as a readable table
    results/plots/*.png            accuracy vs. change strength, and drop by augmentation
"""
import json
import re

import matplotlib.pyplot as plt
import pandas as pd

from config import AUGMENTATIONS, MODELS, RESULTS_DIR


def load_eval():
    rows = []
    for f in sorted((RESULTS_DIR / "eval").glob("*.json")):
        r = json.loads(f.read_text())
        for name, s in r["sets"].items():
            rows.append({"model": r["model"], "aug": r["aug"], "seed": r["seed"], "test_set": name,
                         "acc": s["acc"], "drop": s["drop"],
                         **{f"acc_{c}": v for c, v in s["per_class"].items()}})
    return pd.DataFrame(rows)


def mean_std_table(df, value):
    g = df.groupby(["model", "aug", "test_set"])[value].agg(["mean", "std", "count"]).reset_index()
    g["aug"] = pd.Categorical(g["aug"], AUGMENTATIONS, ordered=True)
    g["model"] = pd.Categorical(g["model"], MODELS, ordered=True)
    return g.sort_values(["model", "aug", "test_set"])


def wide(g):
    g = g.copy()
    g["cell"] = g.apply(lambda r: f"{r['mean'] * 100:.1f} ± {0 if pd.isna(r['std']) else r['std'] * 100:.1f}", axis=1)
    t = g.pivot_table(index=["model", "aug"], columns="test_set", values="cell", aggfunc="first", observed=True)
    return t[["clean"] + [c for c in t.columns if c != "clean"]]


def strength_plot(acc, prefix, xlabel, out):
    sub = acc[acc["test_set"].str.startswith(prefix) | (acc["test_set"] == "clean")].copy()
    if sub.empty:
        return
    neutral = 1.0 if prefix == "brightness" else 0.0
    sub["x"] = sub["test_set"].apply(lambda s: neutral if s == "clean" else float(re.sub(r"^\D+_", "", s)))
    models = [m for m in MODELS if m in set(sub["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        for aug in AUGMENTATIONS:
            s = sub[(sub["model"] == m) & (sub["aug"] == aug)].sort_values("x")
            if len(s):
                ax.errorbar(s["x"], s["mean"] * 100, yerr=s["std"].fillna(0) * 100, marker="o", capsize=3, label=aug)
        ax.set_title(m)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(title="training augmentation", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def drop_bar_plot(drop, out):
    groups = {"background": ["background"], "lighting": None, "rotation": None, "phone": ["phone"]}
    d = drop[drop["test_set"] != "clean"].copy()
    d["group"] = d["test_set"].apply(lambda s: "lighting" if s.startswith(("brightness", "contrast", "color"))
                                     else "rotation" if s.startswith("rotate") else s)
    d = d[d["group"].isin(groups)]
    if d.empty:
        return
    t = d.groupby(["model", "aug", "group"], observed=True)["mean"].mean().reset_index()
    models = [m for m in MODELS if m in set(t["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        p = t[t["model"] == m].pivot(index="aug", columns="group", values="mean") * 100
        p = p.reindex([a for a in AUGMENTATIONS if a in p.index])
        p.plot.bar(ax=ax, rot=30)
        ax.set_title(m)
        ax.set_ylabel("accuracy drop (percentage points)")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    df = load_eval()
    if df.empty:
        raise SystemExit("No results in results/eval. Train and evaluate some models first.")
    acc, drop = mean_std_table(df, "acc"), mean_std_table(df, "drop")
    acc.to_csv(RESULTS_DIR / "summary_accuracy.csv", index=False)
    drop.to_csv(RESULTS_DIR / "summary_drop.csv", index=False)
    per_class = df[df["test_set"] == "clean"].groupby(["model", "aug"])[[c for c in df if c.startswith("acc_")]].mean()

    md = ["# Results (mean ± std over seeds, in %)", "", "## Accuracy", "", wide(acc).to_markdown(), "",
          "## Accuracy drop (clean − modified)", "", wide(drop).to_markdown(), "",
          "## Per-class accuracy on the clean test set", "", (per_class * 100).round(1).to_markdown(), ""]

    cams = [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / "gradcam").glob("*.json"))]
    if cams:
        c = pd.DataFrame(cams)
        cols = [k for k in ["share_on_object", "object_area", "share_minus_area", "sanity_rank_corr"] if k in c]
        md += ["## Grad-CAM share on object", "", c.groupby(["model", "aug"])[cols].mean().round(3).to_markdown(), ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(md))

    plots = RESULTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    strength_plot(acc, "brightness", "brightness factor (1.0 = unchanged)", plots / "brightness.png")
    strength_plot(acc, "rotate", "rotation angle (degrees)", plots / "rotation.png")
    drop_bar_plot(drop, plots / "drop_by_augmentation.png")
    print(f"Wrote {RESULTS_DIR / 'summary.md'} and plots in {plots}")


if __name__ == "__main__":
    main()
