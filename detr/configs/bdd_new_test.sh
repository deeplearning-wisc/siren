#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup/dataset/my_xfdu/detr_out/exps/bdd
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file bdd \
 --dataset bdd \
  --epochs 20 \
  --lr_drop 16 \
   --eval_every 1 \
   --batch_size 1 \
     --load_backbone dino \
   ${PY_ARGS} \



#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset coco_ood_val \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#   ${PY_ARGS} \

