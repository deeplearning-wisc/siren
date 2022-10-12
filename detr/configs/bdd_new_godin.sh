#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup-slow/dataset/my_xfdu/detr_out/exps/bdd_godin
PY_ARGS=${@:1}

#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file bdd \
# --dataset bdd \
#  --epochs 30 \
#  --lr_drop 24 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --godin \
#   ${PY_ARGS} \


#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset coco_ood_val_bdd \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --godin \
#   ${PY_ARGS} \
#
python -u main.py --output_dir ${EXP_DIR} \
    --dataset_file coco \
   --dataset openimages_ood_val \
  --epochs 50 \
  --lr_drop 40 \
   --eval_every 10 \
   --batch_size 1 \
     --load_backbone dino \
     --godin \
   ${PY_ARGS} \

