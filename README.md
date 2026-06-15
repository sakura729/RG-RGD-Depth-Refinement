# RG-RGD: Real-Time Small-Target RGB-D Depth Refinement

This repository contains the open-source companion code for the manuscript:

**RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation**

The code is intended to support reproducible evaluation of the RGB-D depth-refinement components described in the paper. It includes a public VOID benchmark entry point and an RGB-D/IMU self-supervised training entry point for small-target robotic scenes.

## Scope

This release provides:

- supervised RGB-D depth refinement on the public VOID benchmark;
- RGB-D/IMU self-supervised training for small-target video sequences;
- benefit-driven foveated focusing (BFS-SOFA);
- residual-gated RGB-D depth refinement;
- measurement-anchored sparse-to-dense depth hinting;
- IMU-assisted view-synthesis losses;
- template scripts and documentation for reproduction.

This release does not include:

- private RGB-D/IMU datasets;
- trained model weights;
- field-trial execution statistics;
- hardware-control firmware for the laser or gimbal.

The bench-top prototype code path is provided as workflow integration support. It should not be interpreted as a deployment-level execution-effect claim without larger paired trials.

## Repository Layout

```text
RG-RGD-Depth-Refinement/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt
  configs/
    selfsup_paper_command.txt
    void_paper_command.txt
  docs/
    CODE_ALIGNMENT.md
    DATA_PREPARATION.md
    REPRODUCE_SELFSUP.md
    REPRODUCE_VOID.md
    REPRODUCIBILITY_CHECKLIST.md
  scripts/
    run_selfsup.sh
    run_void.sh
  tools/
    train_rgbd_imu_selfsup.py
    train_void_supervised.py
```

## Main Entry Points

1. `tools/train_void_supervised.py`
   - Supervised RGB-D depth refinement on the public VOID benchmark.
   - Default wrapper: `scripts/run_void.sh`.

2. `tools/train_rgbd_imu_selfsup.py`
   - RGB-D/IMU self-supervised training for small-target video sequences.
   - Default wrapper: `scripts/run_selfsup.sh`.

## Paper-to-Code Mapping

| Paper component | Main code location |
| --- | --- |
| Hybrid RGB-D feature extraction | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Benefit-driven foveated scale head | `BFSHead` |
| Small-object focused cross-attention | `SofaCrossAttention` |
| Residual-gated depth refinement | `RGRGDDepthRefiner.forward()` |
| Measurement-anchored fusion | uncertainty heads and fusion block in `RGRGDDepthRefiner.forward()` |
| Lightweight propagation refinement | `LiteLearnedPropRefiner`, `GaussianBPRefiner`, `UACSPNRefiner` |
| IMU-assisted self-supervised view synthesis | `PoseNet`, `IMUCache`, `warp_src_to_tgt`, and the self-supervised training loop |
| VOID benchmark experiment | `tools/train_void_supervised.py` |
| RGB-D/IMU small-target experiment | `tools/train_rgbd_imu_selfsup.py` |

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n rgrgd python=3.10 -y
conda activate rgrgd
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version when needed. See the official PyTorch installation page for platform-specific commands.

## Dataset Preparation

See `docs/DATA_PREPARATION.md` for expected layouts.

The repository does not redistribute VOID or the self-collected RGB-D/IMU data. Users must download or prepare datasets separately and pass local paths to the scripts.

## Reproduce the VOID Experiment

```bash
bash scripts/run_void.sh /path/to/void_release/void_1500 runs/void_rgrgd
```

The script is a template. Adjust batch size, number of workers, and ViT options according to your hardware.

## Run RGB-D/IMU Self-Supervised Training

```bash
bash scripts/run_selfsup.sh /path/to/london_plane_rgbd_imu runs/selfsup_london_plane
```

If depth frames are not registered to the RGB camera, perform depth-to-color registration before training.

## Optional Local ViT Weights

Both scripts support:

```bash
--vit_local_weights /path/to/weights.safetensors
```

Leave this argument empty to use the initialization supported by `timm` or the fallback CNN stem.

## Reproducibility Notes

- Record the exact dataset split, random seed, GPU type, CUDA version, and PyTorch version.
- Keep the default commands unchanged for paper-aligned reproduction unless reporting ablations.
- Optional YOLO or teacher-mask utilities remain in the code for ablation and debugging, but they are disabled in the default reproduction commands.
- See `docs/REPRODUCIBILITY_CHECKLIST.md` before releasing new results.

## Citation

If you use this repository, please cite the corresponding manuscript:

```bibtex
@article{rgrgd2026,
  title  = {RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation},
  author = {Si, Bowen and Ning, Dayong and Hou, Jiaoyi and Gong, Yongjun and Yi, Ming and Zhang, Fengrui and Liu, Zhilei},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

This code is released under the MIT License. See `LICENSE`.
