# ============================================================================
# OWNER: EVERYONE (helpers, usually no changes needed)
# ============================================================================
"""Small helpers shared by the scripts."""
import json
import random

import numpy as np
from PIL import Image

from config import CLASSES, SPLIT_FILE, TRASHNET_DIR


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon Macs
    return torch.device("cpu")


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_split():
    """Returns {"train": [(relative_path, label), ...], "val": [...], "test": [...]}"""
    if not SPLIT_FILE.exists():
        raise SystemExit(f"{SPLIT_FILE} not found. Run: python src/make_split.py")
    split = json.loads(SPLIT_FILE.read_text())
    return {k: [(p, CLASSES.index(p.split("/")[0])) for p in v] for k, v in split.items()}


def image_path(rel):
    """'glass/glass12.jpg' -> full path inside the TrashNet folder."""
    return TRASHNET_DIR / rel


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
