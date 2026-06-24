# Reproducing the VOID benchmark experiment

This is the default command used by `scripts/run_void.sh` for reproducing the public VOID supervised benchmark experiment.
It does **not** enable optional teacher-mask distillation, so no teacher-mask folder is required for the default reproduction path.

## Default reproduction command

```bash
python tools/train_void_supervised.py \
  --root /path/to/void_release/void_1500 \
  --out_dir runs/void_rgrgd \
  --seed 42 \
  --void_split official \
  --epochs 30 \
  --img_h 480 --img_w 640 \
  --batch_size 6 \
  --lr 2.8e-4 \
  --lr_vit 1.0e-5 \
  --vit_no_pretrained \
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
```

You can run the same default configuration through the wrapper script:

```bash
bash scripts/run_void.sh /path/to/void_release/void_1500 runs/void_rgrgd
```

Adjust `--batch_size` and `--num_workers` for the available GPU/CPU memory. The default command uses `--vit_no_pretrained`, so the run does not depend on external ViT-weight downloads.

## Optional local ViT weights

If local ViT weights are used, pass them explicitly and report the setting:

```bash
--vit_local_weights /path/to/weights.safetensors
```

Do not mix this setting with the default run unless it is clearly labeled as a variant.

## Optional ablation: teacher-mask distillation

Teacher-mask distillation is retained only for ablation and is disabled by default. If teacher masks are prepared, place them in a sibling folder of `image/`, for example:

```text
<scene_id>/
  image/
  yolo_mask_v2/
```

Then add the following arguments to the default command:

```bash
--teacher_enable \
--teacher_subdir yolo_mask_v2 \
--w_mask_distill 0.15
```

This optional setting is not required to reproduce the default VOID benchmark path reported by the released scripts.
