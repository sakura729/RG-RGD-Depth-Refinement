#!/usr/bin/env python
"""Run a compact end-to-end verification of the released code.

The script creates small synthetic VOID-style and RGB-D/IMU-style datasets,
then runs one reduced training epoch for both public entry points. It checks
imports, dataset loading, forward passes, loss computation, backpropagation,
validation, and checkpoint writing. The generated data are for software
verification only and are not used to reproduce the manuscript metrics.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def write_void_dataset(root: Path) -> Path:
    void = root / "void_1500"
    for sub in [
        "data/scene_0001/image",
        "data/scene_0001/sparse_depth",
        "data/scene_0001/validity_map",
        "data/scene_0001/ground_truth",
    ]:
        (void / sub).mkdir(parents=True, exist_ok=True)

    height, width = 80, 96
    yy, xx = np.mgrid[0:height, 0:width]
    base_depth_mm = 1200 + 2 * xx + yy
    mask = (((xx + yy) % 3) != 0).astype(np.uint16) * 255

    for i in range(4):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (xx + i * 10) % 255
        rgb[..., 1] = (yy * 2 + i * 20) % 255
        rgb[..., 2] = 120
        cv2.circle(rgb, (20 + i * 8, 30), 6, (220, 180, 40), -1)
        gt_mm = (base_depth_mm + i * 10).astype(np.uint16)
        sd_mm = (gt_mm * (mask > 0)).astype(np.uint16)
        stem = f"{i:06d}.png"
        cv2.imwrite(str(void / "data/scene_0001/image" / stem), rgb)
        cv2.imwrite(str(void / "data/scene_0001/ground_truth" / stem), gt_mm)
        cv2.imwrite(str(void / "data/scene_0001/sparse_depth" / stem), sd_mm)
        cv2.imwrite(str(void / "data/scene_0001/validity_map" / stem), mask)

    def rel(path: Path) -> str:
        return path.relative_to(void).as_posix()

    for split, ids in [("train", [0, 1]), ("test", [2, 3])]:
        with open(void / f"{split}_image.txt", "w", encoding="utf-8") as f_img, open(
            void / f"{split}_sparse_depth.txt", "w", encoding="utf-8"
        ) as f_sd, open(void / f"{split}_validity_map.txt", "w", encoding="utf-8") as f_vm, open(
            void / f"{split}_ground_truth.txt", "w", encoding="utf-8"
        ) as f_gt:
            for i in ids:
                stem = f"{i:06d}.png"
                f_img.write(rel(void / "data/scene_0001/image" / stem) + "\n")
                f_sd.write(rel(void / "data/scene_0001/sparse_depth" / stem) + "\n")
                f_vm.write(rel(void / "data/scene_0001/validity_map" / stem) + "\n")
                f_gt.write(rel(void / "data/scene_0001/ground_truth" / stem) + "\n")

    return void


def write_rgbd_imu_dataset(root: Path) -> Path:
    seq_root = root / "london_plane_rgbd_imu"
    seq = seq_root / "seq_0001"
    (seq / "cam0").mkdir(parents=True, exist_ok=True)
    (seq / "depth0").mkdir(parents=True, exist_ok=True)

    height, width = 96, 96
    yy, xx = np.mgrid[0:height, 0:width]
    for k, timestamp_us in enumerate([0, 50000, 100000, 150000]):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (xx + k * 8) % 255
        rgb[..., 1] = (yy + 40) % 255
        rgb[..., 2] = 80 + k * 20
        cv2.line(rgb, (20 + k, 10), (40 + k, 80), (180, 150, 40), 2)
        depth_mm = (1300 + xx + yy + k * 5).astype(np.uint16)
        depth_mm[((xx + yy + k) % 7) == 0] = 0
        stem = str(timestamp_us)
        cv2.imwrite(str(seq / "cam0" / f"{stem}.png"), rgb)
        cv2.imwrite(str(seq / "depth0" / f"{stem}.png"), depth_mm)

    with open(seq / "imu.csv", "w", encoding="utf-8") as f:
        f.write("timestamp_us,ax,ay,az,gx,gy,gz\n")
        for timestamp_us in range(0, 150001, 10000):
            f.write(f"{timestamp_us},0,0,9.81,0.001,0.002,0.003\n")

    return seq_root


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[verify] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=Path, default=Path(".release_test_runs"))
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    parser.add_argument("--keep", action="store_true", help="keep synthetic data and outputs")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    void_root = write_void_dataset(work)
    selfsup_root = write_rgbd_imu_dataset(work)
    runs = work / "runs"

    void_cmd = [
        sys.executable,
        "tools/train_void_supervised.py",
        "--root",
        str(void_root),
        "--out_dir",
        str(runs / "void_verify"),
        "--void_split",
        "official",
        "--epochs",
        "1",
        "--img_h",
        "64",
        "--img_w",
        "64",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--depth_scale",
        "1000",
        "--base",
        "8",
        "--patch",
        "4",
        "--heads",
        "2",
        "--win_size",
        "4",
        "--tf_depth",
        "1",
        "--bp_iters",
        "1",
        "--cspn_iters",
        "1",
        "--cspn_hidden",
        "8",
        "--vit_name",
        "vit_tiny_patch16_224",
        "--vit_no_pretrained",
        "--freeze_vit_epochs",
        "0",
        "--so_warmup_epochs",
        "0",
        "--benefit_every",
        "1",
        "--eval_min",
        "0.2",
        "--eval_max",
        "5.0",
        "--no-train_apply_eval_range",
        "--train_log_eval_range",
        "--yolo_mode",
        "none",
        "--tta_steps",
        "0",
    ]
    if args.cpu:
        void_cmd += ["--cpu", "--no_amp"]
    run(void_cmd, repo)

    selfsup_cmd = [
        sys.executable,
        "tools/train_rgbd_imu_selfsup.py",
        "--data_root",
        str(selfsup_root),
        "--out_dir",
        str(runs / "selfsup_verify"),
        "--img_h",
        "64",
        "--img_w",
        "64",
        "--depth_scale",
        "1000",
        "--preprocess",
        "resize",
        "--rgb_width",
        "96",
        "--rgb_height",
        "96",
        "--rgb_fx",
        "80",
        "--rgb_fy",
        "80",
        "--rgb_cx",
        "48",
        "--rgb_cy",
        "48",
        "--frame_stride",
        "1",
        "--pair_mode",
        "sliding",
        "--max_pair_gap_us",
        "60000",
        "--imu_gyro_unit",
        "rad",
        "--use_sd_fill",
        "--limit_pairs",
        "3",
        "--val_ratio",
        "0.34",
        "--split_mode",
        "frame",
        "--base",
        "8",
        "--patch",
        "4",
        "--heads",
        "2",
        "--win_size",
        "4",
        "--tf_depth",
        "1",
        "--vit_name",
        "vit_tiny_patch16_224",
        "--max_depth",
        "5",
        "--so_warmup_epochs",
        "0",
        "--benefit_every",
        "1",
        "--benefit_use_photo",
        "--benefit_photo_w",
        "0.5",
        "--residual_mode",
        "--res_alpha",
        "0.5",
        "--delta_max",
        "0.5",
        "--pose_imu_rot",
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--lr",
        "1e-5",
        "--weight_decay",
        "0",
        "--num_workers",
        "0",
        "--w_photo",
        "0.1",
        "--w_geo",
        "0.1",
        "--w_warp",
        "0.1",
        "--w_meas",
        "0.05",
        "--w_smooth",
        "0.0001",
        "--obs_keep_prob",
        "1.0",
        "--obs_max_frac",
        "1.0",
        "--meas_sparse",
        "--meas_keep_prob",
        "1.0",
        "--meas_max_frac",
        "1.0",
        "--vis_every",
        "0",
    ]
    if args.cpu:
        selfsup_cmd += ["--device", "cpu", "--no_amp"]
    run(selfsup_cmd, repo)

    expected = [
        runs / "void_verify" / "rgrgd_void_best.pth",
        runs / "selfsup_verify" / "ckpt_best.pth",
        runs / "selfsup_verify" / "ckpt_last.pth",
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise RuntimeError("Verification finished but checkpoint files are missing: " + ", ".join(missing))

    print("\n[verify] PASS")
    for p in expected:
        print(f"[verify] {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
