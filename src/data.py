# ============================================================================
# OWNER: SÜLEYMAN - Proposal Steps 2-3 (data loading + the 6 augmentation settings, Goal 3)
# The training augmentation ranges at the top are milder than the test strengths in
# config.py on purpose (test changes stay unseen). Discuss with the team before changing.
# ============================================================================
"""Datasets and the six training augmentation settings (Goal 3)."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

import shifts
from config import CLASSES, IMG_SIZE, MEAN, STD, TEXTURE_DIR
from make_masks import load_mask
from utils import image_path, load_rgb, load_split

# Training ranges are kept milder than the test strengths in config.py,
# so the modified test sets are never seen during training.
TRAIN_BRIGHTNESS = (0.7, 1.3)
TRAIN_CONTRAST = (0.7, 1.3)
TRAIN_SATURATION = (0.7, 1.3)
TRAIN_ROTATION = 10          # degrees, +/-
BACKGROUND_PROB = 0.5


def final_transform(train_crop=False):
    if train_crop:
        resize = [T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                  T.RandomHorizontalFlip()]
    else:
        resize = [T.Resize((IMG_SIZE, IMG_SIZE))]
    return T.Compose(resize + [T.ToTensor(), T.Normalize(MEAN, STD)])


class Augment:
    """Applies one augmentation setting to a PIL image (before resizing)."""

    def __init__(self, setting):
        assert setting in ["none", "standard", "photometric", "geometric", "background", "combined"]
        self.setting = setting
        use = {"photometric", "geometric", "background", "standard"} if setting == "combined" else {setting}
        self.background = "background" in use
        self.photometric = "photometric" in use
        self.geometric = "geometric" in use
        self.crop = "standard" in use
        self.textures = sorted((TEXTURE_DIR / "train").glob("*.jpg")) if self.background else []
        if self.background and not self.textures:
            raise SystemExit("No training textures. Run: python src/get_textures.py")
        self.color = T.ColorJitter(brightness=TRAIN_BRIGHTNESS, contrast=TRAIN_CONTRAST,
                                   saturation=TRAIN_SATURATION)
        self.final = final_transform(train_crop=self.crop)

    def __call__(self, img, mask):
        if self.background and random.random() < BACKGROUND_PROB:
            img = shifts.replace_background(img, mask, load_rgb(random.choice(self.textures)))
        if self.geometric:
            angle = random.uniform(-TRAIN_ROTATION, TRAIN_ROTATION)
            img = shifts.rotate(img, angle, shifts.background_color(img, mask))
        if self.photometric:
            img = self.color(img)
        return self.final(img)


class TrashNetSplit(Dataset):
    """One part of the fixed split ("train", "val" or "test")."""

    def __init__(self, part, augment=None, needs_mask=False):
        self.items = load_split()[part]
        self.augment = augment
        self.needs_mask = needs_mask or (augment is not None and (augment.background or augment.geometric))
        self.plain = final_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = load_rgb(image_path(rel))
        if self.augment is None:
            return self.plain(img), label
        mask = load_mask(rel, img.size) if self.needs_mask else None
        return self.augment(img, mask), label


class FolderSet(Dataset):
    """Images stored as <root>/<class>/<file> (modified test sets and phone photos)."""

    def __init__(self, root, only=None):
        root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
        self.items = []
        for label, c in enumerate(CLASSES):
            for p in sorted((root / c).glob("*")):
                if p.suffix.lower() in exts and (only is None or f"{c}/{p.name}" in only):
                    self.items.append((p, label))
        self.plain = final_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        return self.plain(load_rgb(p)), label


def oversampling_sampler(items):
    """Draw rare classes (like 'trash') as often as common ones."""
    labels = np.array([y for _, y in items])
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(labels), replacement=True)


def make_loader(ds, batch_size, shuffle=False, sampler=None, workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available())
