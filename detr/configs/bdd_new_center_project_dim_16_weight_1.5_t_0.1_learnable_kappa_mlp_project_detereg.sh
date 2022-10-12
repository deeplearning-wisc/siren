#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup/dataset/my_xfdu/detr_out/exps/bdd_center_project_dim_16_weight_1.5_t_0.1_learnable_kappa_mlp_project_detreg64
PY_ARGS=${@:1}

#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file bdd \
# --dataset bdd \
#  --epochs 60 \
#  --lr_drop 48 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --pretrain /nobackup/dataset/my_xfdu/detr_out/exps/DETReg_top30_in/checkpoint.pth \
#     --center_loss \
#     --center_loss_scheme_project 1 \
#     --project_dim 64 \
#     --center_weight 1.5 \
#     --unknown_start_epoch 0 \
#     --center_revise \
#     --center_vmf_learnable_kappa \
#     --mlp_project \
#   ${PY_ARGS} \



#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file coco \
# --dataset coco_ood_val_bdd \
#  --epochs 60 \
#  --lr_drop 48 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --pretrain /nobackup/dataset/my_xfdu/detr_out/exps/DETReg_top30_in/checkpoint.pth \
#     --center_loss \
#     --center_loss_scheme_project 1 \
#     --project_dim 64 \
#     --center_weight 1.5 \
#     --unknown_start_epoch 0 \
#     --center_revise \
#     --center_vmf_learnable_kappa \
#     --mlp_project \
#   ${PY_ARGS} \


python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
 --dataset openimages_ood_val \
  --epochs 60 \
  --lr_drop 48 \
   --eval_every 10 \
   --batch_size 1 \
     --pretrain /nobackup/dataset/my_xfdu/detr_out/exps/DETReg_top30_in/checkpoint.pth \
     --center_loss \
     --center_loss_scheme_project 1 \
     --project_dim 64 \
     --center_weight 1.5 \
     --unknown_start_epoch 0 \
     --center_revise \
     --center_vmf_learnable_kappa \
     --mlp_project \
   ${PY_ARGS} \


