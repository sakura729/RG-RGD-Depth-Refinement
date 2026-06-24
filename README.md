# RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Robotics%20(MDPI)-2ea44f.svg)](#citation)

This repository contains the reference implementation for the manuscript:

> RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation  
> Bowen Si, Dayong Ning, Jiaoyi Hou, Yongjun Gong, Ming Yi, Fengrui Zhang, Zhilei Liu  
> Manuscript submitted to *Robotics* (MDPI), 2026

RG-RGD denotes residual-gated RGB-D depth refinement. The released code supports the public VOID benchmark experiment and the RGB-D/IMU self-supervised training pipeline used for the small-target robotic perception study.

## Release Scope

Included:

- Supervised RGB-D depth refinement on the public VOID benchmark.
- RGB-D/IMU self-supervised training for small-target video sequences.
- Reference implementations of BFS-SOFA, residual-gated Bayesian measurement fusion, UACSPN refinement, and IMU-assisted view-synthesis training.
- Reproduction scripts, dataset-layout notes, citation metadata, and reproducibility checklists.

Not included:

- VOID dataset files.
- Private London plane fruit-ball RGB-D/IMU sequences.
- Model checkpoints or local ViT weights.
- Laser-head or gimbal firmware.
- Deployment-level field-trial statistics.

The prototype-related code path is provided to show how refined local geometry can be consumed by a robotic workflow. Dataset files and hardware-control firmware are not redistributed in this repository.

## Repository Layout

```text
RG-RGD-Depth-Refinement/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- OPEN_SOURCE_MANIFEST.md
|-- environment.yml
|-- requirements.txt
|-- requirements-optional.txt
|-- configs/
|   |-- void_paper_command.txt
|   `-- selfsup_paper_command.txt
|-- docs/
|   |-- CODE_ALIGNMENT.md
|   |-- DATA_PREPARATION.md
|   |-- REPRODUCE_VOID.md
|   |-- REPRODUCE_SELFSUP.md
|   `-- REPRODUCIBILITY_CHECKLIST.md
|-- scripts/
|   |-- run_void.sh
|   |-- run_selfsup.sh
|   `-- verify_release.py
`-- tools/
    |-- train_void_supervised.py
    `-- train_rgbd_imu_selfsup.py
```

## Installation

Create the reference conda environment:

```bash
conda env create -f environment.yml
conda activate rgrgd
```

Alternatively, install the Python dependencies manually:

```bash
conda create -n rgrgd python=3.10 -y
conda activate rgrgd
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version when using the manual route. Optional YOLO-based utilities are disabled in the default reproduction commands; install them only when running ablations:

```bash
pip install -r requirements-optional.txt
```

## Dataset Preparation

This repository does not redistribute datasets. Users should prepare datasets locally and pass paths to the scripts.

- VOID: download the official VOID release and point `--root` to a density folder such as `void_release/void_1500`.
- RGB-D/IMU sequences: use the layout documented in `docs/DATA_PREPARATION.md`. Private London plane sequences are available from the corresponding author upon reasonable request, subject to institutional approval.

## Reproducing the Main Experiments

### VOID benchmark

```bash
bash scripts/run_void.sh /path/to/void_release/void_1500 runs/void_rgrgd
```

The default VOID command uses `--vit_no_pretrained` and does not download external ViT weights. Local ViT weights can be supplied with `--vit_local_weights`; report that setting separately from the default run.

### RGB-D/IMU self-supervised training

```bash
bash scripts/run_selfsup.sh /path/to/london_plane_rgbd_imu runs/selfsup_london_plane
```

Depth frames should be registered to the RGB camera before training when the RGB-D sensor stores depth in a different camera frame.

Full commands are recorded in:

- `configs/void_paper_command.txt`
- `configs/selfsup_paper_command.txt`

## Paper-to-Code Mapping

| Manuscript component | Main code location |
| --- | --- |
| Hybrid RGB-D feature extraction | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Dense depth hint from valid measurements | depth-hint preprocessing in both training scripts |
| Benefit-driven foveated scale head | `BFSHead` |
| Small-object focused attention | `SofaCrossAttention` |
| Self-play benefit supervision | two-pass training loop in `tools/train_rgbd_imu_selfsup.py` |
| Residual-gated depth prediction | `RGRGDDepthRefiner.forward()` |
| Bayesian measurement fusion | uncertainty heads and fusion block in `RGRGDDepthRefiner.forward()` |
| UACSPN propagation | `LiteLearnedPropRefiner`, `GaussianBPRefiner`, `UACSPNRefiner` |
| IMU-assisted pose decomposition | `PoseNet`, `IMUCache` |
| View-synthesis warping | `warp_src_to_tgt` |
| VOID benchmark experiment | `tools/train_void_supervised.py` |
| RGB-D/IMU self-supervised experiment | `tools/train_rgbd_imu_selfsup.py` |

Additional notes are available in `docs/CODE_ALIGNMENT.md`.

## Verification

Before sharing a modified version, run:

```bash
python -m py_compile tools/train_void_supervised.py tools/train_rgbd_imu_selfsup.py
```

To run an end-to-end release check without external datasets:

```bash
python scripts/verify_release.py
```

The verification script creates small synthetic VOID-style and RGB-D/IMU-style datasets, runs one reduced training epoch for each entry point, and checks checkpoint writing. It verifies software execution only and is not used to reproduce manuscript metrics.

When reporting new results, record the exact command, git commit hash or release tag, dataset split, seed, GPU, CUDA version, PyTorch version, and whether local ViT weights or optional YOLO utilities were used. A checklist is provided in `docs/REPRODUCIBILITY_CHECKLIST.md`.

## Reported Results

The manuscript reports the following headline values under the paper protocol:

| Setting | Metric | Value |
| --- | --- | --- |
| VOID benchmark | MAE | 24.95 mm |
| VOID benchmark | iMAE | 10.85 |
| Self-collected ROI ablation | ROI geometric error reduction | 15.3% with BFS-SOFA |
| Runtime at 320 x 320 | model / end-to-end latency | 44.57 ms / 72.70 ms mean |

See the manuscript for the complete comparison tables, ablation settings, and runtime protocol.

## Citation

Please cite the associated manuscript when using this code:

```bibtex
@article{si2026rgrgd,
  title   = {RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation},
  author  = {Si, Bowen and Ning, Dayong and Hou, Jiaoyi and Gong, Yongjun and
             Yi, Ming and Zhang, Fengrui and Liu, Zhilei},
  journal = {Robotics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Machine-readable citation metadata are provided in `CITATION.cff`.

## License

This code is released under the MIT License. The VOID dataset and third-party assets retain their own licenses; users are responsible for complying with those terms.
