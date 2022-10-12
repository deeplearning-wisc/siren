#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup/dataset/my_xfdu/detr_out/exps/bdd_unknown_separate_weight_2
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file bdd \
 --dataset bdd \
  --epochs 30 \
  --lr_drop 24 \
   --eval_every 10 \
   --batch_size 1 \
     --load_backbone dino \
     --unknown \
     --separate \
     --separate_loss_weight 2.0 \
     --start_epoch 0 \
     --sample_number 200 \
   ${PY_ARGS} \



#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset coco_ood_val \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#    --unknown \
#     --separate \
#     --separate_loss_weight 2.0 \
#   ${PY_ARGS} \

#python -u main.py --output_dir ${EXP_DIR} \
#    --dataset_file coco \
#   --dataset openimages_ood_val \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#    --unknown \
#     --separate \
#     --separate_loss_weight 2.0 \
#   ${PY_ARGS} \

