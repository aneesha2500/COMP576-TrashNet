# Improving the Robustness of CNN Waste Classifiers with Data Augmentation

ELEC/COMP 576 final project: Aneesha Nayak, Süleyman Inaltekin, Vishnu Muthyalu.

We train a small CNN (from scratch) and ResNet-18 (pretrained) on TrashNet, then test how much accuracy
drops when the background, lighting or rotation changes, and on our own phone photos. We compare six
training augmentations and use Grad-CAM to check whether the models look at the object or the background.

## Who does what

Every file starts with an `OWNER:` comment saying whose it is and what to check. Proposal steps:

| Step | What | Owner | Files | Done when |
|---|---|---|---|---|
| 1 | Data, split, masks, textures, modified test sets | **Aneesha** (masks fixed by **Vishnu**) | `get_data.py`, `make_split.py`, `make_masks.py`, `get_textures.py`, `shifts.py`, `make_test_shifts.py` | Split matches class proportions; 50 random edited images look clean |
| 2 | Baselines (no augmentation) | **Süleyman** | `models.py`, `train.py`, `evaluate.py` | ResNet-18 close to published TrashNet accuracy [15, 16] |
| 3 | Augmentation experiments (2 models × 6 augmentations × 3 seeds) | **Süleyman** | `data.py`, `train.py`, `scripts/run_all.sh` | Accuracy drop measured on every test set |
| 4a | Phone photos (≥20 per class) | **Aneesha** | `data/phone_photos/` | ≥20 photos per class committed |
| 4b | Background segmentation with SAM + Grad-CAM | **Vishnu** | `make_masks_sam.py`, `gradcam.py` | Grad-CAM passes the randomization check [14] |
| 5 | Analysis, tables, plots | **Süleyman** | `analyze.py` | `results/summary.md` + plots ready for the poster |
| – | Poster (Dec 9) and report (Dec 15) | **Everyone** | | |

Order matters: Step 1 has to be done (and `splits/split.json` committed) before anyone trains.

## Folder layout

```
src/                     all code (run from the repo root: python src/<file>.py)
scripts/run_all.sh       runs every experiment
notebooks/colab_runner.ipynb   the same steps as Colab cells
splits/split.json        fixed 70/15/15 split (committed, everyone uses it)
data/phone_photos/       our phone photos (committed)
data/masks_fixed/        SAM-fixed masks (committed)
data/dataset-resized/    TrashNet          (NOT committed, downloaded by get_data.py)
data/masks/, textures/, test_shifts/       (NOT committed, made by the scripts)
checkpoints/             trained models    (NOT committed, too big; keep on the shared Google Drive)
results/                 JSON results, summary tables and plots (committed)
```

## Running it

### On Google Colab (recommended, free GPU)
Open `notebooks/colab_runner.ipynb` in Colab (File > Open notebook > GitHub, or upload it), set
Runtime > Change runtime type > T4 GPU, and run the cells in order.

### On your own computer
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/get_data.py            # TrashNet -> data/dataset-resized
python src/make_split.py          # only the first time (then commit splits/split.json)
python src/make_masks.py --check  # masks + results/mask_check.png
python src/get_textures.py        # DTD textures (or --procedural if the download fails)
python src/make_test_shifts.py    # modified test sets + results/shift_examples.png

python src/train.py --model resnet18 --aug none --seed 0     # trains, then evaluates
bash scripts/run_all.sh                                       # every run
python src/gradcam.py checkpoints/resnet18_none_s0.pt --sanity
python src/analyze.py             # results/summary.md and results/plots/
```
Training on a laptop CPU is slow (the small CNN is OK, ResNet-18 is better on Colab).

## Design requirements (from the proposal)
- Everything trains on one GPU (Colab).
- One fixed, stratified 70/15/15 split: `splits/split.json`. Don't regenerate it.
- Test backgrounds (`data/textures/test`) and test strengths (in `config.py`) are never used in training.
  Training augmentation ranges in `data.py` are milder on purpose.
- Every experiment uses 3 seeds and is reported as mean ± std (`analyze.py` does this).

## Working together with git
```bash
git pull                       # ALWAYS before you start
# ... work ...
git add <files you changed>
git commit -m "Short message: what you did"
git pull --rebase              # get teammates' changes first
git push
```
Only edit your own files when you can. If you need to change someone else's file, tell them first.
Never commit `data/dataset-resized/`, `checkpoints/` or other big files (`.gitignore` already blocks them).
