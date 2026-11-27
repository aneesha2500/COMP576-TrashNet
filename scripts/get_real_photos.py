# ============================================================================
# OWNER: ANEESHA - real-world test photos (Proposal Step 4)
# Instead of taking our own phone photos, we use real-world litter photos from
# TACO (Proenca & Simoes, 2020; http://tacodataset.org, images CC BY 4.0).
# We crop one object per photo and sort it into the 6 TrashNet classes.
# Run on your own computer (needs internet):  python scripts/get_real_photos.py
# Output: data/phone_photos/<class>/taco_<image id>_<ann id>.jpg  + data/phone_photos/CREDITS.csv
# (The folder keeps its old name so evaluate.py picks it up as the "phone" test set.)
# ============================================================================
import csv, io, json, random, sys, urllib.request
from pathlib import Path
from PIL import Image, ImageOps

PER_CLASS = 25          # photos per class
MIN_SIDE = 250          # skip objects smaller than this (pixels in the original photo)
MARGIN = 0.15           # extra space around the object
ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"

# TACO category id -> our class (only clear single-material objects)
MAP = {
    "cardboard": [17, 14, 15, 19],          # corrugated, other carton, egg carton, pizza box
    "glass":     [6, 26, 23],               # glass bottle, glass jar, glass cup
    "metal":     [12, 10, 11, 0, 52],       # drink can, food can, aerosol, foil, scrap metal
    "paper":     [33, 30, 32, 34],          # normal, magazine, wrapping paper, paper bag
    "plastic":   [5, 4, 21, 43, 45, 47],    # plastic bottles, plastic cup, plastic containers
    "trash":     [42, 39, 22, 46, 31],      # crisp packet, wrapper, foam cup, foam container, tissues
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "phone_photos"

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def main():
    random.seed(0)
    print("Downloading TACO annotations...")
    d = json.loads(get(ANN_URL))
    imgs = {im["id"]: im for im in d["images"]}
    credits = []
    for cls, cats in MAP.items():
        (OUT / cls).mkdir(parents=True, exist_ok=True)
        cands = [a for a in d["annotations"] if a["category_id"] in cats
                 and min(a["bbox"][2], a["bbox"][3]) >= MIN_SIDE]
        random.shuffle(cands)
        used_imgs, n = set(), 0
        for a in cands:
            if n >= PER_CLASS:
                break
            if a["image_id"] in used_imgs:      # one object per photo, for variety
                continue
            im = imgs[a["image_id"]]
            try:
                img = Image.open(io.BytesIO(get(im["flickr_url"])))
                t = ImageOps.exif_transpose(img)
                img = t if t.size == (im["width"], im["height"]) else img
                if img.size != (im["width"], im["height"]):
                    continue
            except Exception as e:
                print(f"  skip {im['flickr_url']}: {e}")
                continue
            x, y, w, h = a["bbox"]
            s = max(w, h) * (1 + 2 * MARGIN)        # square crop around the object
            cx, cy = x + w / 2, y + h / 2
            box = [max(0, cx - s / 2), max(0, cy - s / 2), min(img.width, cx + s / 2), min(img.height, cy + s / 2)]
            crop = img.convert("RGB").crop([int(v) for v in box])
            crop.thumbnail((800, 800))
            name = f"taco_{im['id']}_{a['id']}.jpg"
            crop.save(OUT / cls / name, quality=88)
            credits.append([cls, name, im["file_name"], im["flickr_url"]])
            used_imgs.add(a["image_id"]); n += 1
            print(f"  {cls}: {n}/{PER_CLASS}", end="\r")
        print(f"  {cls}: {n} photos          ")
    with open(OUT / "CREDITS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "taco_file", "source_url"])
        w.writerows(credits)
    print(f"Done. Look through data/phone_photos/ and delete any photo that is in the wrong class or unclear.")
    print("Source: TACO, Proenca & Simoes 2020, http://tacodataset.org (CC BY 4.0)")

if __name__ == "__main__":
    sys.exit(main())
