#!/usr/bin/env bash
set -euo pipefail

VOID_ROOT="${1:-/path/to/void_release/void_1500}"
OUT_DIR="${2:-runs/void_rgrgd}"

python tools/train_void_supervised.py \
  --root "$VOID_ROOT" \
  --out_dir "$OUT_DIR" \
  --seed 42 \
  --void_split official \
  --epochs 30 \
  --img_h 480 --img_w 640 \
  --batch_size 6 \
  --lr 2.8e-4 \
  --lr_vit 1.0e-5 \
  --lr_warmup_epochs 2 \
  --lr_min_ratio 0.05 \
  --wd 1e-4 \
  --freeze_vit_epochs 2 \
  --so_warmup_epochs 5 \
  --bp_iters 3 \
  --cspn_iters 4 --cspn_hidden 24 \
  --min_sigma 0.02 \
  --max_sigma 5.0 \
  --max_depth 10.0 \
  --w_nll_max 0.05 \
  --nll_warmup_epochs 15 \
  --benefit_every 4 \
  --benefit_beta 0.15 \
  --w_benefit 0.10 \
  --w_mass 0.30 \
  --w_edgetv 0.015 \
  --w_scale 0.04 \
  --yolo_mode none \
  --tta_steps 0 \
  --eval_min 0.2 --eval_max 5.0 \
  --train_log_eval_range \
  --no-train_apply_eval_range \
  --ema_enable --ema_decay 0.9999 --ema_eval \
  --num_workers 8
