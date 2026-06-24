# Code Alignment with the Manuscript

This document maps the main RG-RGD manuscript components to the released code.

## Default Reproduction Paths

- `scripts/run_void.sh` runs the public VOID supervised benchmark experiment.
- `scripts/run_selfsup.sh` runs the RGB-D/IMU self-supervised experiment used for the small-target robotic perception study.

Both scripts disable optional YOLO-based online masks by default. The default VOID command also uses `--vit_no_pretrained`, so reproduction does not require external ViT-weight downloads.

## Component Mapping

| Manuscript component | Code location |
| --- | --- |
| Hybrid RGB-D feature extraction | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Dense depth hint from valid measurements | depth preprocessing and filled-depth input construction in both training scripts |
| Benefit-driven foveated scale head | `BFSHead` |
| Small-object focused attention | `SofaCrossAttention` |
| Self-play benefit supervision | baseline/focused two-pass loop in `tools/train_rgbd_imu_selfsup.py` |
| Residual-gated depth prediction | `RGRGDDepthRefiner.forward()` |
| Bayesian measurement fusion | uncertainty heads and variance-weighted fusion in `RGRGDDepthRefiner.forward()` |
| UACSPN propagation | `LiteLearnedPropRefiner`, `GaussianBPRefiner`, `UACSPNRefiner` |
| IMU-assisted pose decomposition | `PoseNet`, `IMUCache`, `integrate_gyro_between()` |
| View-synthesis warping | `warp_src_to_tgt()` |
| VOID benchmark training/evaluation | `tools/train_void_supervised.py` |
| RGB-D/IMU self-supervised training | `tools/train_rgbd_imu_selfsup.py` |

## Optional Utilities

Some utilities remain available for ablation studies:

- YOLO-based masks can be used as pseudo-mask priors when explicitly enabled.
- Teacher-mask distillation can be enabled for controlled VOID ablations.
- Local ViT weights can be supplied with `--vit_local_weights`.

These optional paths are disabled in the default reproduction commands. Any result using them should be labeled as a variant rather than as the default run.
