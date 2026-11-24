#!/usr/bin/env bash
# ============================================================================
# OWNER: SÜLEYMAN - Proposal Step 3 (every model x augmentation x seed)
# 2 models x 6 augmentations x 3 seeds = 36 runs. ResNet-18 runs take ~5-10 min each
# on a Colab GPU; the small CNN is faster. Split the runs between Colab sessions with
#   bash scripts/run_all.sh resnet18      or      bash scripts/run_all.sh smallcnn
# Finished runs are skipped (only evaluated again if their results are missing), so you can
# restart after a disconnect. An interrupted run leaves a .partial file and is retrained.
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
        if [ ! -f "results/eval/${m}_${a}_s${s}.json" ]; then
          echo "evaluate ${m}_${a}_s${s} (trained earlier)"
          python src/evaluate.py "checkpoints/${m}_${a}_s${s}.pt"
        else
          echo "skip ${m}_${a}_s${s} (already done)"
        fi
        continue
      fi
      python src/train.py --model "$m" --aug "$a" --seed "$s"
    done
  done
done
python src/analyze.py
