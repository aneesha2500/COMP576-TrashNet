#!/usr/bin/env bash
# ============================================================================
# OWNER: SÜLEYMAN - Proposal Step 3 (every model x augmentation x seed)
# 2 models x 6 augmentations x 3 seeds = 36 runs. ResNet-18 runs take ~5-10 min each
# on a Colab GPU; the small CNN is faster. Split the runs between Colab sessions with
#   bash scripts/run_all.sh resnet18      or      bash scripts/run_all.sh smallcnn
# Runs whose checkpoint already exists are skipped, so you can restart after a disconnect.
# ============================================================================
set -e
cd "$(dirname "$0")/.."
MODELS=${1:-"smallcnn resnet18"}
AUGS=${AUGS:-"none standard photometric geometric background combined"}
SEEDS=${SEEDS:-"0 1 2"}
for m in $MODELS; do
  for a in $AUGS; do
    for s in $SEEDS; do
      if [ -f "checkpoints/${m}_${a}_s${s}.pt" ]; then
        echo "skip ${m}_${a}_s${s} (already done)"; continue
      fi
      python src/train.py --model "$m" --aug "$a" --seed "$s"
    done
  done
done
python src/analyze.py
