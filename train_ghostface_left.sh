#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/GhostFaceNets'
DATA_DIR='/mnt/data/afarec/data/PetFace'

echo "Start training for GhostFaceNet with loss arcface for class all"
PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
  python $ROOT_DIR/main.py \
  --output "${ROOT_DIR}/work_dir/arcface_all" \
  --loss "arcface" \
  --data "${DATA_DIR}/split/all/train.csv"

echo "Start training for GhostFaceNet with loss cosface for class cat"
PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
  python $ROOT_DIR/main.py \
  --output "${ROOT_DIR}/work_dir/cosface_cat" \
  --loss "cosface" \
  --data "${DATA_DIR}/split/cat/train.csv"