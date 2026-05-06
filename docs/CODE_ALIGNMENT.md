# Code alignment with the manuscript

This repository provides the official implementation of the experiments described in the manuscript.

## Default reproduction path

- `scripts/run_void.sh` runs the public VOID supervised benchmark experiment without optional teacher-mask distillation.
- `scripts/run_selfsup.sh` runs the RGB-D/IMU self-supervised experiment using the benefit-driven BFS-SOFA path.

## Optional utilities

Some optional utilities remain in the source for ablation/debugging:

- YOLO-based masks can be used as pseudo-mask priors for ablation.
- Teacher-mask distillation can be enabled in the VOID script for ablation.

These optional paths are disabled in the default reproduction commands so that the released default behavior matches the manuscript's benefit-driven/self-supervised description.

## Naming note

The self-supervised script uses `RGRGDDepthRefiner` as the public model class. A backward-compatible alias, `SODR_ViT_YOLO_Transformer`, is kept only to avoid breaking older internal checkpoints or scripts.
