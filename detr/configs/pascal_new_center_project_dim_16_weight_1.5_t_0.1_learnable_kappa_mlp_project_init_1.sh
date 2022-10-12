#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup/dataset/my_xfdu/detr_out/exps/pascal_center_project_dim_16_weight_1.5_t_0.1_learnable_kappa_mlp_project_init_1
PY_ARGS=${@:1}

#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file voc \
# --dataset voc \
#  --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --center_loss \
#     --center_loss_scheme_project 1 \
#     --project_dim 16 \
#     --center_weight 1.5 \
#     --unknown_start_epoch 0 \
#     --center_revise \
#     --center_vmf_learnable_kappa \
#     --mlp_project \
#     --learnable_kappa_init 1 \
#   ${PY_ARGS} \



#python -u main.py --output_dir ${EXP_DIR} \
#      --dataset_file coco \
# --dataset coco_ood_val \
# --epochs 50 \
#  --lr_drop 40 \
#   --eval_every 10 \
#   --batch_size 1 \
#     --load_backbone dino \
#     --center_loss \
#     --center_loss_scheme_project 1 \
#     --project_dim 16 \
#     --center_weight 1.5 \
#     --unknown_start_epoch 0 \
#     --center_revise \
#     --center_vmf_learnable_kappa \
#     --mlp_project \
#     --learnable_kappa_init 1 \
#   ${PY_ARGS} \


python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
 --dataset openimages_ood_val \
 --epochs 50 \
  --lr_drop 40 \
   --eval_every 10 \
   --batch_size 1 \
     --load_backbone dino \
     --center_loss \
     --center_loss_scheme_project 1 \
     --project_dim 16 \
     --center_weight 1.5 \
     --unknown_start_epoch 0 \
     --center_revise \
     --center_vmf_learnable_kappa \
     --mlp_project \
     --learnable_kappa_init 1 \
   ${PY_ARGS} \


