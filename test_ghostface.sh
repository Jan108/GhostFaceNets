#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/GhostFaceNets'
DATA_DIR='/mnt/data/afarec/data/PetFace'

for loss in "arcface" "cosface"; do
  loss_a="A"
  if [ $loss == "cosface" ]; then
      loss_a="C"
  fi
  for cls in "bird" "cat" "dog" "small_animals"; do
    echo "Start training for GhostFaceNet with loss ${loss} for class ${cls}"
    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
      python $ROOT_DIR/evaluation.py \
        --output "${ROOT_DIR}/work_dir_b256/${loss}_${cls}" \
        --weights "${ROOT_DIR}/work_dir_b256/${loss}_${cls}/ghostV2-1.3-1-(${loss_a})_basic_model_latest.h5" \
        --img_path "${DATA_DIR}/images" \
        --img_verification "${DATA_DIR}/split/${cls}/verification.csv" \
        --img_identification "${DATA_DIR}/split/${cls}/identification_img.csv"
  done
  PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
    python $ROOT_DIR/evaluation.py \
    --output "${ROOT_DIR}/work_dir/${loss}_all" \
    --weights "${ROOT_DIR}/work_dir/${loss}_all/ghostV2-1.3-1-(${loss_a})_basic_model_latest.h5" \
    --img_path "${DATA_DIR}/images" \
    --img_verification "${DATA_DIR}/split/all/verification.csv" \
    --img_identification "${DATA_DIR}/split" \
    --ident-general
done
