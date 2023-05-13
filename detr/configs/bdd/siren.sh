#!/usr/bin/env bash

set -x

EXP_DIR=./snapshots/bdd/siren
PY_ARGS=${@:2}


if [ "$1" = "bdd_id" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
        --dataset_file bdd \
   --dataset bdd \
    --epochs 30 \
    --lr_drop 24 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
       --project_dim 64 \
       --vmf_weight 1.5 \
      --siren \
      --siren_evaluate \
     ${PY_ARGS}
elif [ "$1" = "coco_ood" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
   --dataset coco_ood_val_bdd \
    --epochs 30 \
    --lr_drop 24 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
       --project_dim 64 \
       --vmf_weight 1.5 \
      --siren \
      --siren_evaluate \
   ${PY_ARGS}
elif [ "$1" = "openimages_ood" ]; then
  python -u main.py --output_dir ${EXP_DIR} \
      --dataset_file coco \
     --dataset openimages_ood_val \
   --epochs 30 \
    --lr_drop 24 \
     --eval_every 10 \
     --batch_size 1 \
       --load_backbone dino \
       --project_dim 64 \
       --vmf_weight 1.5 \
      --siren \
      --siren_evaluate \
      --eval_bdd \
   ${PY_ARGS}
fi

