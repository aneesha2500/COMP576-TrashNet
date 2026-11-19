# ============================================================================
# OWNER: EVERYONE (shared settings)
# Change settings here, not inside other files, so all three of us use the same values.
# Aneesha: test-set strengths (BRIGHTNESS_LEVELS, ROTATION_ANGLES, ...).
# Süleyman: MODELS / AUGMENTATIONS / SEEDS used in the experiments.
# ============================================================================
"""Shared settings used by every script. Change paths here if your folders are different."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRASHNET_DIR = DATA_DIR / "dataset-resized"      # unzipped TrashNet
MASK_DIR = DATA_DIR / "masks"                    # object masks (white = object), regenerated each session
MASK_FIX_DIR = DATA_DIR / "masks_fixed"          # Vishnu's SAM fixes; committed to git, used first
TEXTURE_DIR = DATA_DIR / "textures"              # background textures (train/ and test/)
SHIFT_DIR = DATA_DIR / "test_shifts"             # modified test sets
PHONE_DIR = DATA_DIR / "phone_photos"            # our own photos, one folder per class

SPLIT_FILE = ROOT / "splits" / "split.json"      # fixed train/val/test split (committed to git)
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224

# ImageNet mean/std, used by both models so inputs look the same
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Strengths for the modified test sets (Goal 2)
BRIGHTNESS_LEVELS = [0.4, 0.6, 1.4, 1.8]        # 1.0 = unchanged
CONTRAST_LEVELS = [0.5, 1.5]
COLOR_CASTS = {"warm": (1.15, 1.0, 0.8), "cool": (0.8, 1.0, 1.15)}   # RGB multipliers
ROTATION_ANGLES = [15, 30, 45, 90]

MODELS = ["smallcnn", "resnet18"]
AUGMENTATIONS = ["none", "standard", "photometric", "geometric", "background", "combined"]
SEEDS = [0, 1, 2]
