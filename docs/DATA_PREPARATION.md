# Data preparation

## VOID benchmark

`tools/train_void_supervised.py` accepts `--root` pointing to a VOID density folder, for example:

```text
/path/to/void_release/void_1500
```

The directory should contain either the official split files:

```text
train_image.txt
train_sparse_depth.txt
train_validity_map.txt
train_ground_truth.txt
test_image.txt
test_sparse_depth.txt
test_validity_map.txt
test_ground_truth.txt
```

or a `data/` folder with per-scene subfolders:

```text
void_1500/
  data/
    <scene_id>/
      image/ or rgb/
      sparse_depth/
      validity_map/
      ground_truth/
```

Depth values are expected to be stored as unsigned 16-bit PNGs. Use `--depth_scale` to convert raw integer values to meters. The default for VOID is `256.0`.

Optional teacher masks can be placed in a sibling folder of `image/`, for example:

```text
<scene_id>/
  image/
  yolo_mask_v2/
```

Then run with:

```bash
--teacher_enable --teacher_subdir yolo_mask_v2
```

## RGB-D/IMU self-supervised dataset

The self-supervised script accepts either a root directory containing sequences or explicit directories.

Recommended root layout:

```text
london_plane_rgbd_imu/
  seq_0001/
    cam0/       # RGB frames
    depth0/     # depth frames aligned to RGB if possible
    imu.csv     # timestamped IMU measurements
  seq_0002/
    cam0/
    depth0/
    imu.csv
```

Alternative explicit arguments:

```bash
--rgb_dir /path/to/cam0 \
--depth_dir /path/to/depth0 \
--imu_csv /path/to/imu.csv
```

### Important notes

- RGB, depth, and IMU timestamps should be synchronized or close enough for frame pairing.
- If depth is captured in a different camera frame, perform depth-to-color registration before training.
- Use `--depth_scale 1000` for millimeter depth PNGs and `--depth_scale 256` for VOID-style depth PNGs.
- Use `--imu_gyro_unit rad` if gyroscope values are in rad/s and `--imu_gyro_unit deg` if they are in deg/s.
