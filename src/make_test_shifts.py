# ============================================================================
# OWNER: ANEESHA - Proposal Step 1 / Goal 2 (modified test sets)
# Success check from the proposal: look at results/shift_examples.png and 50 random
# images in data/test_shifts/ and make sure the edits look clean.
# Re-run this after Vishnu fixes masks with SAM.
# ============================================================================
"""
Build the modified versions of the TEST set (Goal 2).

    python src/make_test_shifts.py

Writes data/test_shifts/<shift_name>/<class>/<image>.jpg, for example:
    background            object pasted on a held-out texture (data/textures/test)
    brightness_0.4 ...    darker / brighter
    contrast_0.5 ...      lower / higher contrast
    color_warm, color_cool
    rotate_15 ... rotate_90
It also saves results/shift_examples.png (one image under every change) for a visual check.
Everything is made with a fixed seed so all of us get the same test sets.
"""
import random

import numpy as np
from PIL import Image

import shifts
from config import (BRIGHTNESS_LEVELS, COLOR_CASTS, CONTRAST_LEVELS, RESULTS_DIR,
                    ROTATION_ANGLES, SHIFT_DIR, TEXTURE_DIR)
from make_masks import load_mask
from utils import image_path, load_rgb, load_split


def all_shifts():
    """name -> function(img, mask, rng) returning the changed image."""
    textures = sorted((TEXTURE_DIR / "test").glob("*.jpg"))
    if not textures:
        raise SystemExit("No test textures. Run: python src/get_textures.py")
    out = {"background": lambda img, m, rng: shifts.replace_background(img, m, load_rgb(rng.choice(textures)))}
    for b in BRIGHTNESS_LEVELS:
        out[f"brightness_{b}"] = lambda img, m, rng, b=b: shifts.change_brightness(img, b)
    for c in CONTRAST_LEVELS:
        out[f"contrast_{c}"] = lambda img, m, rng, c=c: shifts.change_contrast(img, c)
    for name, rgb in COLOR_CASTS.items():
        out[f"color_{name}"] = lambda img, m, rng, rgb=rgb: shifts.color_cast(img, rgb)
    for a in ROTATION_ANGLES:
        out[f"rotate_{a}"] = lambda img, m, rng, a=a: shifts.rotate(img, a, shifts.background_color(img, m))
    return out


def main():
    test = load_split()["test"]
    funcs = all_shifts()
    examples = {}
    for name, f in funcs.items():
        rng = random.Random(0)
        for rel, _ in test:
            img = load_rgb(image_path(rel))
            mask = load_mask(rel, img.size)
            new = f(img, mask, rng)
            out = SHIFT_DIR / name / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            new.save(out, quality=95)
            if rel == test[0][0]:
                examples[name] = new
        print(f"{name}: {len(test)} images")
    save_examples(load_rgb(image_path(test[0][0])), examples)


def save_examples(original, examples):
    import matplotlib.pyplot as plt
    items = [("original", original)] + list(examples.items())
    cols = 5
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, im) in zip(axes.flat, items):
        ax.imshow(im)
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "shift_examples.png", dpi=100)
    print(f"Saved {RESULTS_DIR / 'shift_examples.png'}")


if __name__ == "__main__":
    main()
