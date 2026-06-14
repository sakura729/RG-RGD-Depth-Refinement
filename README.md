# RG-RGD: Real-Time Small-Target RGB-D Depth Refinement

> **Manuscript status:** This code is directly associated with a manuscript submitted to *Journal of Real-Time Image Processing*. If you use this repository, please cite the corresponding manuscript.

This repository contains the open-source implementation for the manuscript:

**RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation**

The code provides two reproducible entry points:

1. `tools/train_void_supervised.py` — supervised RGB-D depth refinement on the public VOID benchmark.
2. `tools/train_rgbd_imu_selfsup.py` — RGB-D/IMU self-supervised training for small-target robotic video sequences.

## Main components

- BFS-SOFA benefit-driven foveated focusing for task-relevant small-target regions.
- RGB-guided residual-gated depth refinement.
- Measurement-anchored sparse-to-dense depth hinting and refinement.
- IMU-assisted self-supervised view-synthesis training.
- VOID benchmark training/evaluation pipeline.
- Real-time small-target RGB-D refinement for robotic laser-ablation workflows.

## Paper-to-code mapping

| Paper component | Main code location |
| --- | --- |
| Hybrid RGB-D feature extraction | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Benefit-driven foveated scale head | `BFSHead` |
| Small-object focused cross-attention | `SofaCrossAttention` |
| Residual-gated depth refinement | `RGRGDDepthRefiner.forward()` / residual output path |
| Measurement-anchored fusion | uncertainty heads and fusion block in `RGRGDDepthRefiner.forward()` |
| GBPN-lite / belief-propagation-style refinement | `LiteLearnedPropRefiner` / `GaussianBPRefiner` |
| Uncertainty-aware CSPN refinement | `UACSPNRefiner` |
| IMU-assisted self-supervised view synthesis | `PoseNet`, `IMUCache`, `warp_src_to_tgt`, training loop in `tools/train_rgbd_imu_selfsup.py` |
| VOID benchmark experiment | `tools/train_void_supervised.py` |
| London plane RGB-D/IMU experiment | `tools/train_rgbd_imu_selfsup.py` |

The default reproduction scripts use the RG-RGD / BFS-SOFA depth-refinement path. Optional YOLO or teacher-mask utilities may remain in the code for ablation and debugging, but they are not required by the default commands.

## Installation

```bash
conda create -n rgrgd python=3.10 -y
conda activate rgrgd
pip install -r requirements.txt
