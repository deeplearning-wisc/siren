#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup-slow/dataset/my_xfdu/detr_out/exps/DETReg_top30_in100
PY_ARGS=${@:1}

python -u main.py \
--output_dir ${EXP_DIR} \
--dataset imagenet100 \
--strategy topk \
--load_backbone swav \
--max_prop 30 \
--object_embedding_loss \
--lr_backbone 0 \
--epochs 5 \
--lr_drop 4 \
${PY_ARGS}