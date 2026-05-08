# RG-RGD: Benefit-Driven Self-Supervised Depth Refinement

> **Manuscript status:** This code is directly associated with a manuscript submitted to *Machine Vision and Applications*. If you use this repository, please cite the corresponding manuscript.

This repository contains the open-source implementation for the manuscript:

**Benefit-Driven Self-Supervised Depth Refinement for Precise Small-Target 3D Localization in Robotic Vision**

The code provides two reproducible entry points:

1. `tools/train_void_supervised.py` — supervised RGB-D depth refinement on the public VOID benchmark.
2. `tools/train_rgbd_imu_selfsup.py` — RGB-D/IMU self-supervised training for small-target video sequences.

## Main components

- BFS-SOFA benefit-driven foveated focusing for small-target regions.
- RGB-guided residual-gated depth refinement.
- Measurement-anchored sparse-to-dense depth hinting and refinement.
- IMU-assisted self-supervised view-synthesis training.
- VOID benchmark training/evaluation pipeline.


## Paper-to-code mapping

| Paper component | Main code location |
| --- | --- |
| Hybrid RGB-D feature extraction | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Benefit-driven foveated scale head | `BFSHead` |
| Small-object focused cross-attention | `SofaCrossAttention` |
| Residual-gated depth refinement | `RGRGDDepthRefiner.forward()` / residual output path |
| Measurement-anchored Bayesian fusion | uncertainty heads and fusion block in `RGRGDDepthRefiner.forward()` |
| GBPN-lite / belief-propagation-style refinement | `LiteLearnedPropRefiner` / `GaussianBPRefiner` |
| Uncertainty-aware CSPN refinement | `UACSPNRefiner` |
| IMU-assisted self-supervised view synthesis | `PoseNet`, `IMUCache`, `warp_src_to_tgt`, training loop in `tools/train_rgbd_imu_selfsup.py` |
| VOID benchmark experiment | `tools/train_void_supervised.py` |
| London plane RGB-D/IMU experiment | `tools/train_rgbd_imu_selfsup.py` |

The default reproduction scripts use the benefit-driven/self-supervised BFS-SOFA path. Optional YOLO or teacher-mask utilities remain in the code for ablation and debugging, but they are not required by the default commands.

## Installation

```bash
conda create -n rgrgd python=3.10 -y
conda activate rgrgd
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version when needed. See the official PyTorch installation page for platform-specific commands.

## Dataset preparation

See `docs/DATA_PREPARATION.md` for expected dataset layouts.

## Reproduce the VOID experiment

```bash
bash scripts/run_void.sh /path/to/void_release/void_1500 runs/void_rgrgd
```

The script is a template. Adjust batch size, number of workers, and ViT options according to your hardware.

## Run RGB-D/IMU self-supervised training

```bash
bash scripts/run_selfsup.sh /path/to/london_plane_rgbd_imu runs/selfsup_london_plane
```

If your depth frames are not registered to the RGB camera, perform depth-to-color registration before training.

## Optional local ViT weights

Both scripts support `--vit_local_weights /path/to/weights.safetensors`. Leave this argument empty if you want to use the model initialization supported by `timm` or the fallback CNN stem.

## Citation

If you use this repository, please cite the corresponding manuscript:

```bibtex
@article{rgrgd2026,
  title   = {Benefit-Driven Self-Supervised Depth Refinement for Precise Small-Target 3D Localization in Robotic Vision},
  author  = {Si, Bowen and Ning, Dayong and Hou, Jiaoyi and Gong, Yongjun and Yi, Ming and Zhang, Fengrui and Liu, Zhilei},
  journal = {Manuscript submitted to The Visual Computer},
  year    = {2026}
}
```

## License

This code is released under the MIT License. See `LICENSE`.
