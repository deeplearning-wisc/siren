#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup-slow/dataset/my_xfdu/detr_out/exps/pascal_new_detreg
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} \
    --dataset_file voc \
   --dataset voc \
  --epochs 100 \
  --lr_drop 70 \
   --eval_every 10 \
   --batch_size 1 \
   --pretrain /nobackup-slow/dataset/my_xfdu/detr_out/exps/DETReg_top30_in/checkpoint.pth \
   ${PY_ARGS}




#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset coco_ood_val \
#  --epochs 100 \
#  --lr_drop 70 \
#   --eval_every 10 \
#   --batch_size 1 \
#   --pretrain /nobackup-slow/dataset/my_xfdu/detr_out/exps/DETReg_top30_in/checkpoint.pth \
#   ${PY_ARGS} \

