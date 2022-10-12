#!/usr/bin/env bash

set -x

EXP_DIR=./snapshots/voc/vanilla
PY_ARGS=${@:2}

if [ "$1" = "voc_train" ] || [ "$1" = "voc_val" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
        --dataset_file voc \
   --dataset voc \
    --epochs 50 \
    --lr_drop 40 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
     ${PY_ARGS} \
elif [ "$1" = "coco_ood" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
     --dataset coco_ood_val \
    --epochs 50 \
    --lr_drop 40 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
     ${PY_ARGS} \
elif [ "$1" = "openimages_ood" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
     --dataset openimages_ood_val \
    --epochs 50 \
    --lr_drop 40 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
     ${PY_ARGS} \

