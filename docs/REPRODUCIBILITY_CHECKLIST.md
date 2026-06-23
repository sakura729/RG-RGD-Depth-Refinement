# Reproducibility Checklist

Use this checklist before reporting results from this repository.

## Environment

- Record operating system, GPU model, CUDA version, Python version, PyTorch version, and `timm` version.
- Record whether local ViT weights were used.
- Record random seed and deterministic settings.
- Record the git commit hash or release tag.

## VOID Benchmark

- Record the VOID density setting and split files used.
- Record `--depth_scale`, `--eval_min`, and `--eval_max`.
- Keep `scripts/run_void.sh` unchanged for the paper-aligned default run.
- Report MAE, RMSE, iMAE, and iRMSE together.
- State whether EMA evaluation or test-time adaptation was enabled.

## RGB-D/IMU Self-Supervised Training

- Record camera intrinsics and whether depth is registered to RGB.
- Record timestamp synchronization policy and maximum frame-pair gap.
- Record IMU gyroscope unit and preprocessing mode.
- Report whether the split is scene-based or frame-based.
- Save qualitative outputs and training logs for audit.

## Ablations

- Clearly label any run that enables optional YOLO masks, teacher-mask distillation, local ViT weights, or non-default loss weights.
- Do not mix ablation results with default reproduction results in the same table without labels.

## Prototype Integration

- Treat prototype execution as workflow integration unless a paired and statistically designed outdoor trial is reported.
- Record hardware configuration, target distance, lighting, and environmental disturbances if prototype results are reported.

## Archiving

- Save the exact command line.
- Save `requirements.txt`, `environment.yml`, or `pip freeze`.
- Save raw logs and generated tables with the manuscript materials.
- If possible, create a GitHub release and archive it with a DOI before final publication.
