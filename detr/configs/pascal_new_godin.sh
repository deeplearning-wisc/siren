#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup/dataset/my_xfdu/detr_out/exps/pascal_godin
PY_ARGS=${@:1}

#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file voc \
# --dataset voc \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --godin \
#   ${PY_ARGS} \

#
#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset coco_ood_val \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --godin \
#   ${PY_ARGS} \

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

