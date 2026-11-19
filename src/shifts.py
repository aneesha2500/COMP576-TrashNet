# ============================================================================
# OWNER: ANEESHA - the image changes (background, lighting, rotation)
# Used by make_test_shifts.py (test sets) and by data.py (Süleyman's training augmentations).
# ============================================================================
"""Image changes used for both the modified test sets and the training augmentations."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def replace_background(img, mask, texture):
    """Put the object (mask True) on top of a texture. All are PIL/np at the same size."""
    texture = texture.resize(img.size)
    soft = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return Image.composite(img, texture, soft)


def change_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)


def change_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)


def color_cast(img, rgb):
    a = np.asarray(img).astype(np.float32) * np.array(rgb, dtype=np.float32)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def background_color(img, mask):
    """Median color of the background, used to fill corners after rotating."""
    a = np.asarray(img)
    bg = a[~mask] if (~mask).any() else a.reshape(-1, 3)
    return tuple(int(v) for v in np.median(bg, axis=0))


def rotate(img, angle, fill):
    return img.rotate(angle, fillcolor=fill)
