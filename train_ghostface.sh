#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/GhostFaceNets'
DATA_DIR='/mnt/data/afarec/data/PetFace'

for loss in "arcface" "cosface"; do
  for cls in "all" "bird" "cat" "dog" "small_animals"; do
    echo "Start training for GhostFaceNet with loss ${loss} for class ${cls}"
    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
      python $ROOT_DIR/main.py \
        --output "${ROOT_DIR}/work_dir/${loss}_${cls}" \
        --loss $loss \
        --data "${DATA_DIR}/split/${cls}/train.csv"
  done
done
