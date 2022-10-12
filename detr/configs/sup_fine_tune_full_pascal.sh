#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup-slow/dataset/my_xfdu/detr_out/exps/supervised_fine_tune_full_pascal
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} \
    --dataset_file voc \
  --dataset voc \
  --epochs 100 \
  --lr_drop 70 \
   --eval_every 10 \
   --batch_size 1 \
   --load_backbone supervised \
   ${PY_ARGS}

#   --dataset_file coco \
#  --dataset coco_ood_val \