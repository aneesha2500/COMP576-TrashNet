# ============================================================================
# OWNER: ANEESHA - Proposal Step 1 (data preparation)
# Download TrashNet and check there are 2,527 images.
# ============================================================================
"""
Download TrashNet into data/dataset-resized.

    python src/get_data.py

If the download fails (or you already have the folder), unzip dataset-resized.zip
yourself so you get data/dataset-resized/<class>/*.jpg
"""
import io
import shutil
import urllib.request
import zipfile

from config import CLASSES, DATA_DIR, TRASHNET_DIR

URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def main():
    if all((TRASHNET_DIR / c).exists() for c in CLASSES):
        print(f"TrashNet already at {TRASHNET_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading TrashNet (~40 MB)...")
    data = urllib.request.urlopen(URL).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(DATA_DIR)
    junk = DATA_DIR / "__MACOSX"
    if junk.exists():
        shutil.rmtree(junk)
    n = sum(len(list((TRASHNET_DIR / c).glob("*.jpg"))) for c in CLASSES)
    print(f"Done: {n} images in {TRASHNET_DIR}")   # should be 2527


if __name__ == "__main__":
    main()
