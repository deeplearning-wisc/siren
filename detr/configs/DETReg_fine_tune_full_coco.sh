#!/usr/bin/env bash

set -x

EXP_DIR=/nobackup-slow/dataset/my_xfdu/detr_out/exps/DETReg_fine_tune_full_coco_in100_5epoch
PY_ARGS=${@:1}

python -u main.py \
--output_dir ${EXP_DIR} \
--dataset coco \
--pretrain /nobackup-slow/dataset/my_xfdu/detr_out/exps/DETReg_top30_in100/checkpoint.pth \
${PY_ARGS}