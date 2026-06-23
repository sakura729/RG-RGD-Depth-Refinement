# Open-Source Manifest

This manifest lists the files included in the RG-RGD code-availability release.

## Root Files

- `README.md`: overview, installation, usage, scope, citation, and repository layout.
- `LICENSE`: MIT License.
- `CITATION.cff`: citation metadata for GitHub and archival services.
- `OPEN_SOURCE_MANIFEST.md`: this release inventory.
- `environment.yml`: reference conda environment.
- `requirements.txt`: core Python dependencies for the released scripts.
- `requirements-optional.txt`: optional dependencies for ablation utilities.
- `.gitignore`: excludes datasets, checkpoints, logs, and local environments.
- `.gitattributes`: keeps scripts and source files with consistent line endings.

## Configuration Files

- `configs/void_paper_command.txt`: paper-aligned VOID command template.
- `configs/selfsup_paper_command.txt`: paper-aligned RGB-D/IMU command template.

## Documentation

- `docs/CODE_ALIGNMENT.md`: mapping between manuscript components and code locations.
- `docs/DATA_PREPARATION.md`: expected layouts for VOID and RGB-D/IMU data.
- `docs/REPRODUCE_VOID.md`: VOID benchmark reproduction instructions.
- `docs/REPRODUCE_SELFSUP.md`: RGB-D/IMU self-supervised reproduction instructions.
- `docs/REPRODUCIBILITY_CHECKLIST.md`: checklist for reporting reproducible results.

## Scripts

- `scripts/run_void.sh`: shell wrapper for the public VOID supervised experiment.
- `scripts/run_selfsup.sh`: shell wrapper for RGB-D/IMU self-supervised training.
- `scripts/run_smoke_test.py`: synthetic end-to-end smoke test for both training entry points.

## Source Files

- `tools/train_void_supervised.py`: supervised VOID benchmark training/evaluation script.
- `tools/train_rgbd_imu_selfsup.py`: RGB-D/IMU self-supervised training script.

## Not Included

- VOID dataset files.
- Private London plane RGB-D/IMU data.
- Model checkpoints or local ViT weights.
- Laser-head or gimbal hardware firmware.
- Deployment-level field-trial statistics.

## Evidence Boundary

The prototype-related code path documents workflow integration with target localization, branch screening, cutting-point back-projection, and gimbal-based execution. It is not packaged as hardware-control firmware or as deployment-level field-trial evidence.
