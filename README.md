# RG-RGD: Residual-Gated RGB-D Depth Refinement for Robotic Laser Ablation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Robotics%20(MDPI)-2ea44f.svg)](#citation)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](#)

Companion code for the manuscript:

> **RG-RGD: Real-Time Small-Target RGB-D Depth Refinement for Robotic Laser Ablation**
> Bowen Si, Dayong Ning, Jiaoyi Hou, Yongjun Gong, Ming Yi, Fengrui Zhang, Zhilei Liu
> Naval Architecture and Ocean Engineering College, Dalian Maritime University
> Submitted to *Robotics* (MDPI), 2026

This repository releases the open-source reference implementation used to generate the depth-refinement results reported in the paper. It is intended to support reproducible benchmark evaluation and self-supervised training in small-target robotic perception scenes.

---

## Highlights

- **Task-oriented local depth refinement.** Reallocates the error budget toward task-relevant regions instead of optimizing only image-wide metrics.
- **Self-play benefit-driven foveation (BFS-SOFA).** Resolves the circular dependency between the focus mask and the focused prediction without manual region labels.
- **Residual-gated Bayesian measurement fusion (BMF).** Predicts a bounded residual around the dense depth hint and fuses it with the raw observation through variance-weighted reasoning.
- **Edge-aware UACSPN propagation.** Suppresses cross-boundary diffusion through an RGB edge barrier, ROI-boosted gating, and per-step measurement re-anchoring.
- **IMU-assisted self-supervised training.** Decomposes inter-frame pose into gyroscope-anchored rotation and visually estimated translation for stable view-synthesis supervision.

### Reported Results

| Setting | Metric | Value |
| --- | --- | --- |
| VOID benchmark | MAE | **24.95 mm** (lowest among compared methods) |
| VOID benchmark | iMAE | **10.85** (lowest among compared methods) |
| Self-collected ROI ablation | ROI geometric error reduction | **−15.3%** with BFS-SOFA |
| Runtime (RTX 3070 Ti, 320×320) | Model latency / End-to-end | 44.57 ms / 72.70 ms (mean) |

See the manuscript for full comparison tables, ablation, and runtime breakdown.

---

## Scope of This Release

**Included**

- Supervised RGB-D depth refinement on the public VOID benchmark.
- RGB-D / IMU self-supervised training for small-target video sequences.
- Reference implementations of:
  - Hybrid RGB-D feature extraction with a ViT-based RGB stem and a depth-hint stem.
  - Benefit-driven foveated scale head and small-object focused cross-attention (BFS-SOFA).
  - Residual-gated depth refinement with predicted per-pixel uncertainty.
  - Bayesian measurement fusion (BMF) anchored to raw observations.
  - Uncertainty-aware convex spatial propagation (UACSPN) with RGB edge barrier and validity re-injection.
  - IMU-assisted PoseNet and view-synthesis warping for self-supervised training.
- Template scripts and reproduction documentation.

**Not included**

- Private RGB-D / IMU sequences of London plane fruit balls.
- Trained model checkpoints.
- Quantitative field-trial statistics.
- Hardware control firmware for the laser head and gimbal.

The robotic prototype path is provided as workflow integration support. As emphasized in the paper, the laser-ablation use case is **demonstrative**, illustrating how refined local geometry feeds downstream physical rules. It should not be interpreted as a deployment-level performance claim without larger paired outdoor trials.

---

## Repository Layout

```text
RG-RGD-Depth-Refinement/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── selfsup_paper_command.txt   # exact command for self-supervised paper run
│   └── void_paper_command.txt      # exact command for VOID paper run
├── docs/
│   ├── CODE_ALIGNMENT.md           # paper section ↔ code module mapping
│   ├── DATA_PREPARATION.md         # expected dataset layouts
│   ├── REPRODUCE_VOID.md           # step-by-step VOID reproduction
│   ├── REPRODUCE_SELFSUP.md        # step-by-step self-supervised reproduction
│   └── REPRODUCIBILITY_CHECKLIST.md
├── scripts/
│   ├── run_void.sh                 # wrapper for the VOID experiment
│   └── run_selfsup.sh              # wrapper for the self-supervised experiment
└── tools/
    ├── train_void_supervised.py    # entry point for VOID
    └── train_rgbd_imu_selfsup.py   # entry point for self-supervised training
```

---

## Paper-to-Code Mapping

The table below maps each component described in the manuscript to its main code location. A more detailed mapping, including equation-level references, is available in `docs/CODE_ALIGNMENT.md`.

| Paper component | Section | Main code location |
| --- | --- | --- |
| Hybrid RGB-D feature extraction | §2.1, Eq. (3) | `ViTSRGBStem`, `rgb_local`, `dep_stem` in `tools/train_*` |
| Dense depth hint from valid measurements | §2.1, Eq. (2) | depth-hint generation in preprocessing |
| Benefit-driven foveated scale head | §2.2, Eqs. (4)–(5) | `BFSHead` |
| Small-object focused cross-attention | §2.2 | `SofaCrossAttention` |
| Self-play benefit supervision | §2.2, Algorithm 1 | two-pass training loop in `tools/train_rgbd_imu_selfsup.py` |
| Residual-gated depth prediction | §2.3, Eq. (6) | `RGRGDDepthRefiner.forward()` |
| Bayesian measurement fusion (BMF) | §2.3, Eq. (7) | uncertainty heads and fusion block in `RGRGDDepthRefiner.forward()` |
| UACSPN propagation | §2.4, Eqs. (8)–(10) | `LiteLearnedPropRefiner`, `GaussianBPRefiner`, `UACSPNRefiner` |
| IMU-assisted pose decomposition | §2.5, Eqs. (11)–(12) | `PoseNet`, `IMUCache` |
| View-synthesis warping | §2.5, Eq. (13) | `warp_src_to_tgt` |
| Composite training objective | §2.5, Eq. (14) | self-supervised training loop |
| VOID benchmark experiment | §3.2 | `tools/train_void_supervised.py` |
| Self-collected small-target experiment | §3.3 | `tools/train_rgbd_imu_selfsup.py` |

---

## Installation

```bash
conda create -n rgrgd python=3.10 -y
conda activate rgrgd
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version separately. See the official PyTorch installation page for platform-specific commands. Experiments in the paper used an NVIDIA GeForce RTX 3070 Ti laptop GPU with automatic mixed precision enabled.

---

## Dataset Preparation

This repository does **not** redistribute datasets. Users prepare datasets locally and pass paths to the scripts.

- **VOID** — download from the official VOID release. The expected layout follows `void_release/void_1500/`.
- **Self-collected London plane RGB-D / IMU sequences** — not publicly redistributed. Available from the corresponding author upon reasonable request, subject to institutional approval.

Refer to `docs/DATA_PREPARATION.md` for the expected directory structure, frame-pair convention, and IMU synchronization requirements.

---

## Quick Start

### 1. Reproduce the VOID experiment

```bash
bash scripts/run_void.sh /path/to/void_release/void_1500 runs/void_rgrgd
```

The script is a paper-aligned template. Adjust batch size, worker count, and ViT options to your hardware. The exact command used in the paper is recorded in `configs/void_paper_command.txt`.

### 2. Run RGB-D / IMU self-supervised training

```bash
bash scripts/run_selfsup.sh /path/to/london_plane_rgbd_imu runs/selfsup_london_plane
```

If depth frames are not registered to the RGB camera, perform depth-to-color registration before training. The exact command used in the paper is recorded in `configs/selfsup_paper_command.txt`.

### 3. Optional ViT weights

Both entry points accept locally cached ViT weights:

```bash
--vit_local_weights /path/to/weights.safetensors
```

Leave the argument empty to fall back to the `timm` initialization or to the convolutional stem.

---

## Reproducibility Notes

To match the paper-reported numbers as closely as possible:

- Use the default commands in `configs/`. Modify them only when reporting ablations or hardware-specific tuning.
- Record the exact dataset split, random seed, GPU model, CUDA version, and PyTorch version.
- BFS-SOFA is enabled in the focused-pass branch only. The baseline pass uses uniform feature weighting, as described in Algorithm 1 of the paper.
- Optional YOLO and teacher-mask utilities remain in the codebase for ablation and debugging. They are **disabled** in the default reproduction commands.
- The full reproducibility checklist is in `docs/REPRODUCIBILITY_CHECKLIST.md`.

Some randomness from cuDNN nondeterminism and mixed-precision rounding is expected. Reported metrics are within typical run-to-run fluctuation when the protocol above is followed.

---

## Limitations

Consistent with the manuscript discussion:

- VOID results show unresolved large-error outliers on RMSE and iRMSE.
- The self-supervised loop depends on IMU-assisted pose constraints and may degrade under wind-induced branch motion, rolling-shutter artifacts, or platform vibration.
- The current Python pipeline reaches ~13.75 FPS at 320×320; sustained video-rate operation would benefit from compiled depth-hint generation and embedded-GPU optimization.
- The robotic laser-ablation use case is demonstrative; quantitative outdoor benchmarks remain future work.

---

## Citation

If you use this code or build on the method, please cite the manuscript:

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

The bibliographic entry will be updated to the final journal reference once the paper is accepted. A machine-readable record is also provided in `CITATION.cff`.

---

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (grant 52571377), the National Key Research and Development Program of China (grant 2023YFC2809804), and the Fundamental Research Funds for the Central Universities (grants 3132023513 and 3132025120). The authors thank the laboratory members who assisted with prototype construction and data collection.

The implementation reuses ideas from prior depth-completion and self-supervised depth literature cited in the manuscript, including NLSPN, CostDCNet, PENet, and the SfMLearner / Monodepth2 self-supervised line of work.

---

## License

This code is released under the MIT License. See `LICENSE`.

The VOID dataset and any third-party assets retain their respective licenses. Users are responsible for complying with the terms of any external dataset or model used in conjunction with this repository.
