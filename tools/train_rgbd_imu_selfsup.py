# -*- coding: utf-8 -*-
r"""
Official training script for the RGB-D/IMU self-supervised experiment in the RG-RGD paper.

This script trains the IMU-assisted self-supervised RG-RGD depth refinement pipeline for
small-target RGB-D video sequences. It supports either a dataset root containing
sequence folders or explicit --rgb_dir, --depth_dir, and --imu_csv arguments.

Example using a dataset root:
    python tools/train_rgbd_imu_selfsup.py \
        --out_dir runs/selfsup_london_plane \
        --data_root /path/to/london_plane_rgbd_imu \
        --img_h 320 --img_w 320 \
        --depth_scale 1000 \
        --imu_gyro_unit rad \
        --frame_stride 2 --pair_mode sliding --max_pair_gap_us 600000 \
        --use_sd_fill --pose_imu_rot \
        --preprocess crop_square \
        --max_depth 10.0 \
        --epochs 30 --batch_size 1 --lr 1.5e-5 \
        --weight_decay 5e-5 \
        --residual_mode --res_alpha 1.0 --delta_max 1.0 \
        --w_photo 1.0 --w_geo 1.0 --geo_photo_beta 2.0 \
        --w_warp 0.8 \
        --meas_sparse --meas_keep_prob 0.7 --meas_max_frac 0.25 \
        --w_meas 0.10 --meas_align scale --meas_loss_type huber --huber_delta 0.10 \
        --w_smooth 0.001 \
        --occ_thresh 0.10 \
        --obs_keep_prob 0.5 --obs_max_frac 0.5 \
        --benefit_use_photo --benefit_photo_w 0.5 \
        --val_ratio 0.2 --split_mode scene \
        --vit_name vit_small_patch16_224_dino \
        --vis_every 50 --vis_max_per_epoch 10 --vis_split both --vis_save_npz

Expected data:
    - RGB frames, depth frames, and IMU timestamps should be time-synchronized.
    - If depth is not aligned to the RGB camera, perform depth-to-color registration first.
    - See docs/DATA_PREPARATION.md for the suggested directory layout.

Notes:
    - Replace all dataset/model paths with your local paths.
    - Optional local ViT weights can be provided with --vit_local_weights.
    - No private data are included in this repository.
"""

import os, glob, random, argparse, time, math, copy, re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import cv2
cv2.setNumThreads(0)

# Optional: safetensors loader (for local ViT weights)
try:
    from safetensors.torch import load_file as safe_load_file  # type: ignore
except Exception:
    safe_load_file = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# -------------------------
# Optional deps
# -------------------------
try:
    import timm
except Exception:
    timm = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# -------------------------
# ViT RGB Stem (global semantics, upsampled to full-res)
# -------------------------
class ViTSRGBStem(nn.Module):
    """RGB stem that extracts global semantics using a timm ViT and upsamples to full resolution.

    If timm is unavailable, falls back to a lightweight CNN stem so the training script still runs.
    """

    def __init__(
        self,
        base: int = 32,
        model_name: str = "vit_small_patch16_224",
        patch: int = 16,
        pretrained: bool = True,
        local_weights: str = "",
    ):
        super().__init__()
        self.base = int(base)
        self.model_name = str(model_name)
        self.patch = max(1, int(patch))
        self.pretrained = bool(pretrained)
        self.local_weights = str(local_weights) if local_weights is not None else ""

        # ImageNet normalization (common for timm ViTs); data is already in [0,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.vit = None
        self._vit_dim = None

        if timm is not None:
            try:
                # Many timm ViTs support dynamic_img_size; try it first.
                try:
                    self.vit = timm.create_model(
                        self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="", dynamic_img_size=True
                    )
                except TypeError:
                    self.vit = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="")
            except Exception as e:
                # timm model name mismatch is common across versions; try a small remap set before giving up.
                if str(self.model_name).lower() in ("dino_vits8", "dinovits8"):
                    tried = []
                    for alt in [
                        "vit_small_patch8_224_dino",
                        "vit_small_patch16_224_dino",
                        "vit_small_patch8_224",
                        "vit_small_patch16_224",
                    ]:
                        tried.append(alt)
                        try:
                            self.vit = timm.create_model(alt, pretrained=self.pretrained, num_classes=0, global_pool="")
                            self.model_name = alt
                            print(f"[vit] remap dino_vits8 -> {alt}")
                            break
                        except Exception:
                            self.vit = None
                    if self.vit is None:
                        print(f"[warn] timm ViT create_model failed ({self.model_name}): {e}. Tried {tried}. Falling back to CNN stem.")
                else:
                    print(f"[warn] timm ViT create_model failed ({self.model_name}): {e}. Falling back to CNN stem.")
                self.vit = None if self.vit is None else self.vit


        if self.vit is not None:
            self._vit_dim = int(getattr(self.vit, "num_features", getattr(self.vit, "embed_dim", self.base)))
            self.proj = nn.Sequential(
                nn.Conv2d(self._vit_dim, self.base, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.base),
                nn.SiLU(inplace=True),
            )
            self._try_load_local_weights(self.local_weights)
        else:
            # fallback: simple CNN stem (keeps interface identical)
            self.proj = nn.Sequential(
                nn.Conv2d(3, self.base, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.base),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.base, self.base, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.base),
                nn.SiLU(inplace=True),
            )

    def _try_load_local_weights(self, pth: str):
        if self.vit is None:
            return
        pth = str(pth or "").strip()
        if not pth:
            return
        if not os.path.isfile(pth):
            print(f"[warn] vit_local_weights not found: {pth} (skip)")
            return

        try:
            if pth.lower().endswith(".safetensors"):
                if safe_load_file is None:
                    raise RuntimeError("safetensors not installed. Please `pip install safetensors` or use a .pth/.pt file.")
                sd = safe_load_file(pth)
            else:
                sd = torch.load(pth, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]

            if not isinstance(sd, dict):
                raise RuntimeError(f"Unexpected checkpoint format: {type(sd)}")

            # strip common prefixes
            cleaned = {}
            for k, v in sd.items():
                kk = k
                for pref in ("module.", "model.", "backbone.", "vit."):
                    if kk.startswith(pref):
                        kk = kk[len(pref) :]
                cleaned[kk] = v

            missing, unexpected = self.vit.load_state_dict(cleaned, strict=False)
            if missing or unexpected:
                print(f"[vit] loaded local weights: missing={len(missing)} unexpected={len(unexpected)}")
            else:
                print("[vit] loaded local weights OK")
        except Exception as e:
            print(f"[warn] failed to load vit_local_weights={pth}: {e}")

    def set_vit_trainable(self, trainable: bool):
        if self.vit is None:
            return
        for p in self.vit.parameters():
            p.requires_grad = bool(trainable)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return (B, base, H, W) feature map."""
        if self.vit is None:
            return self.proj(rgb)

        B, C, H, W = rgb.shape
        x = (rgb - self.mean) / (self.std + 1e-6)

        # ViT forward
        if hasattr(self.vit, "forward_features"):
            feats = self.vit.forward_features(x)
        else:
            feats = self.vit(x)

        # unwrap common containers
        if isinstance(feats, dict):
            feats = feats.get("x", next(iter(feats.values())))
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        # tokens -> feature map
        if feats.dim() == 4:
            fm = feats
        elif feats.dim() == 3:
            tok = feats  # (B, N, D)
            N = tok.shape[1]
            Hp = max(1, H // self.patch)
            Wp = max(1, W // self.patch)

            if N == Hp * Wp + 1:
                tok = tok[:, 1:, :]
                N = tok.shape[1]

            # If shape mismatch, best-effort infer square grid (works for many ViTs)
            if N != Hp * Wp:
                s = int(math.sqrt(N))
                if s * s == N:
                    Hp, Wp = s, s
                else:
                    # fallback: crop/pad to Hp*Wp
                    M = Hp * Wp
                    if N > M:
                        tok = tok[:, :M, :]
                    else:
                        pad = tok.new_zeros((B, M - N, tok.shape[2]))
                        tok = torch.cat([tok, pad], dim=1)

            fm = tok.transpose(1, 2).contiguous().view(B, -1, Hp, Wp)
        else:
            raise RuntimeError(f"Unexpected ViT feature shape: {tuple(feats.shape)}")

        fm = self.proj(fm)
        if fm.shape[-2:] != (H, W):
            fm = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)
        return fm


# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_u16(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(path)
    return x


def read_mask01(path: str) -> np.ndarray:
    """Read a mask image and return float32 in [0,1].

    Supports grayscale / BGR / BGRA. Some masks are saved as single-channel PNGs,
    so we must NOT blindly call cvtColor(BGR2GRAY) when channels==1.

    NOTE:
    - In some environments (e.g., when ultralytics patches cv2.imread to support
      unicode paths), cv2.imread may raise FileNotFoundError instead of returning
      None when the file does not exist. We treat any such error as "missing mask"
      and return None so the caller can safely fall back to zeros.
    """
    try:
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except FileNotFoundError:
        return None
    except Exception:
        # Be conservative: any read error -> missing mask
        return None

    if m is None:
        return None

    # Convert to 2D grayscale safely
    if m.ndim == 3:
        c = m.shape[2]
        try:
            if c == 4:
                m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY)
            elif c == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            elif c == 1:
                m = m[:, :, 0]
            else:
                m = m[:, :, 0]
        except cv2.error:
            # Fallback: take first channel
            m = m[:, :, 0]

    m = m.astype(np.float32)

    # Normalize to [0,1]
    mx = float(m.max()) if m.size else 0.0
    if mx > 1.0:
        denom = 65535.0 if mx > 255.0 else 255.0
        m = m / denom

    m = np.clip(m, 0.0, 1.0)
    return m


def safe_resize(img: np.ndarray, size, interp) -> np.ndarray:
    """Resize image keeping explicit (H,W) if provided.
    Args:
        img: HxW[xC] array
        size: int (square) or (H,W) tuple/list
        interp: cv2 interpolation
    """
    if isinstance(size, (tuple, list)):
        h, w = int(size[0]), int(size[1])
    else:
        h = w = int(size)
    return cv2.resize(img, (w, h), interpolation=interp)

def fill_sparse_depth_nearest(depth: np.ndarray, vm: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill for sparse depth using OpenCV distance transform with labels.

    Args:
        depth: float32 array (H,W), with 0 indicating invalid.
        vm: float/bool array (H,W), where 1 indicates valid depth samples.
    Returns:
        filled: float32 array (H,W), dense depth where invalid pixels are filled from nearest valid pixel.
                If there are no valid pixels, returns a copy of `depth`.
    """
    if depth is None or vm is None:
        return depth.astype(np.float32, copy=True)
    if depth.ndim != 2:
        depth2 = depth.squeeze()
    else:
        depth2 = depth
    if vm.ndim != 2:
        vm2 = vm.squeeze()
    else:
        vm2 = vm

    if vm2.max() < 0.5:
        return depth2.astype(np.float32, copy=True)

    inv = (vm2 < 0.5).astype(np.uint8)  # invalid=1, valid=0
    # distanceTransform expects non-zero as foreground, zeros as background (targets).
    _dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

    ys, xs = np.where(inv == 0)
    if ys.size == 0:
        return depth2.astype(np.float32, copy=True)

    lbls = labels[ys, xs]
    max_lbl = int(labels.max())
    map_y = np.zeros(max_lbl + 1, dtype=np.int32)
    map_x = np.zeros(max_lbl + 1, dtype=np.int32)

    # safeguard label 0 (should not be used, but avoid index errors)
    map_y[0] = int(ys[0])
    map_x[0] = int(xs[0])

    map_y[lbls] = ys.astype(np.int32)
    map_x[lbls] = xs.astype(np.int32)

    filled = depth2[map_y[labels], map_x[labels]]
    return filled.astype(np.float32, copy=False)



def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _extract_scene_id(rgb_path: str) -> str:
    parts = rgb_path.replace("\\", "/").split("/")
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))


def split_pairs_train_val(
    pairs: List[Tuple[str, str, str, str]],
    val_ratio: float,
    split_mode: str,
    seed: int = 42,
):
    rng = random.Random(int(seed))

    if split_mode == "frame":
        pp = list(pairs)
        rng.shuffle(pp)
        n_total = len(pp)
        n_val = max(1, int(n_total * float(val_ratio)))
        val_pairs = pp[:n_val]
        tr_pairs = pp[n_val:]
        info = f"[split] mode=frame | total={n_total} | train_frames={len(tr_pairs)} | val_frames={len(val_pairs)}"
        return tr_pairs, val_pairs, info

    scene2pairs: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for p in pairs:
        sid = _extract_scene_id(p[0])
        scene2pairs.setdefault(sid, []).append(p)

    scenes = sorted(scene2pairs.keys())
    rng.shuffle(scenes)
    n_scenes = len(scenes)
    n_val_scenes = max(1, int(n_scenes * float(val_ratio)))
    val_scenes = set(scenes[:n_val_scenes])

    tr_pairs, val_pairs = [], []
    for sid, plist in scene2pairs.items():
        if sid in val_scenes:
            val_pairs.extend(plist)
        else:
            tr_pairs.extend(plist)

    info = (
        f"[split] mode=scene | scenes total={n_scenes} | val_scenes={len(val_scenes)} | "
        f"train_frames={len(tr_pairs)} | val_frames={len(val_pairs)}"
    )
    return tr_pairs, val_pairs, info


def apply_eval_range(gt_m: torch.Tensor, gtv: torch.Tensor, eval_min_m: float, eval_max_m: float):
    if eval_min_m is None or eval_max_m is None:
        return gtv
    if eval_min_m <= 0 or eval_max_m <= 0:
        return gtv
    v = (gt_m >= eval_min_m) & (gt_m <= eval_max_m)
    return gtv * v.float()


def depth_metrics(mu_m: torch.Tensor, gt_m: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    v = valid > 0.5
    if v.sum() < 10:
        return {"mae_m": 0.0, "rmse_m": 0.0, "absrel": 0.0}
    e = (mu_m[v] - gt_m[v]).abs()
    mae = e.mean().item()
    rmse = torch.sqrt(((mu_m[v] - gt_m[v]) ** 2).mean()).item()
    absrel = (e / gt_m[v].clamp_min(1e-3)).mean().item()
    return {"mae_m": mae, "rmse_m": rmse, "absrel": absrel}


def edge_aware_smoothness(depth_m: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    dx_d = torch.abs(depth_m[:, :, :, 1:] - depth_m[:, :, :, :-1])
    dy_d = torch.abs(depth_m[:, :, 1:, :] - depth_m[:, :, :-1, :])
    dx_i = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    dy_i = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-10.0 * dx_i)
    wy = torch.exp(-10.0 * dy_i)
    return (dx_d * wx).mean() + (dy_d * wy).mean()


def edge_aware_tv(m: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    dx_m = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1])
    dy_m = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :])
    dx_i = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    dy_i = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-10.0 * dx_i)
    wy = torch.exp(-10.0 * dy_i)
    return (dx_m * wx).mean() + (dy_m * wy).mean()




# -------------------------
# Differentiable blur / coherence helpers (for BFS stability)
# -------------------------
def _gaussian_kernel2d(ks: int, sigma: float, device, dtype):
    ks = int(ks)
    if ks <= 1:
        k = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
        return k
    # ensure odd
    if ks % 2 == 0:
        ks += 1
    sigma = float(max(sigma, 1e-6))
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel.view(1, 1, ks, ks)


def gaussian_blur2d(x: torch.Tensor, ks: int = 7, sigma: float = 2.0) -> torch.Tensor:
    """Depthwise gaussian blur for (B,C,H,W). ks<=1 returns x."""
    if ks is None or int(ks) <= 1:
        return x
    if int(ks) % 2 == 0:
        ks = int(ks) + 1
    B, C, H, W = x.shape
    k = _gaussian_kernel2d(int(ks), float(sigma), device=x.device, dtype=x.dtype)
    k = k.repeat(C, 1, 1, 1)  # (C,1,ks,ks)
    pad = int(ks) // 2
    return F.conv2d(x, k, padding=pad, groups=C)


def mask_local_coherence_loss(m: torch.Tensor, image: torch.Tensor, ks: int = 7) -> torch.Tensor:
    """
    Penalize isolated speckles by matching m to its local average,
    while allowing discontinuities on strong RGB edges.
    """
    if ks is None or int(ks) <= 1:
        return m.new_tensor(0.0)
    if int(ks) % 2 == 0:
        ks = int(ks) + 1
    # local mean
    mp = F.avg_pool2d(m, kernel_size=int(ks), stride=1, padding=int(ks)//2)
    # edge weights (high edge => smaller penalty)
    dx_i = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    dy_i = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    edge = torch.zeros_like(m)
    edge[:, :, :, 1:] += dx_i
    edge[:, :, 1:, :] += dy_i
    w = torch.exp(-10.0 * edge)  # smooth away speckles in low-texture areas
    return ((m - mp) ** 2 * w).mean()


def depth_grad_l1(mu: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, image: torch.Tensor, edge_k: float = 2.0) -> torch.Tensor:
    """Encourage sharp geometry by matching depth gradients, weighted by RGB edges."""
    # gradients
    dx_mu = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    dy_mu = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    dx_gt = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    dy_gt = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    # valid for gradient pairs
    vx = valid[:, :, :, 1:] * valid[:, :, :, :-1]
    vy = valid[:, :, 1:, :] * valid[:, :, :-1, :]
    # rgb edge magnitude
    dx_i = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    dy_i = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    wx = (1.0 + float(edge_k) * dx_i).clamp(1.0, 1.0 + float(edge_k))
    wy = (1.0 + float(edge_k) * dy_i).clamp(1.0, 1.0 + float(edge_k))
    lx = (dx_mu - dx_gt).abs() * vx * wx
    ly = (dy_mu - dy_gt).abs() * vy * wy
    return lx.sum() / (vx.sum() + 1e-6) + ly.sum() / (vy.sum() + 1e-6)
def inv_softplus(y: float) -> float:
    # stable inverse softplus for scalar y > 0
    y = float(max(y, 1e-8))
    # softplus(x)=log(1+exp(x)) => x=log(exp(y)-1)
    return float(math.log(math.expm1(y)))


# -------------------------
# VOID scan
# -------------------------
def find_void_pairs_via_scenes(
    void_density_dir: str,
    max_scenes: int,
    max_total: int,
    verbose=True,
    per_scene_cap: int = 0,
    seed: int = 42,
) -> List[Tuple[str, str, str, str]]:
    data_dir = os.path.join(void_density_dir, "data")
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Cannot find data/ under {void_density_dir}")

    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(scenes) == 0:
        raise RuntimeError(f"No scenes under {data_dir}")

    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    pairs: List[Tuple[str, str, str, str]] = []
    for sc in scenes:
        sc_dir = os.path.join(data_dir, sc)

        gt_dir = os.path.join(sc_dir, "ground_truth")
        sd_dir = os.path.join(sc_dir, "sparse_depth")
        vm_dir = os.path.join(sc_dir, "validity_map")

        rgb_dir = os.path.join(sc_dir, "image")
        if not os.path.isdir(rgb_dir):
            alt = os.path.join(sc_dir, "rgb")
            if os.path.isdir(alt):
                rgb_dir = alt

        if not all(os.path.isdir(x) for x in [rgb_dir, gt_dir, sd_dir, vm_dir]):
            if verbose:
                missing = [x for x in [rgb_dir, gt_dir, sd_dir, vm_dir] if not os.path.isdir(x)]
                print(f"[warn] skip scene={sc} missing={missing}")
            continue

        rgb_files = glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg"))
        gt_files = glob.glob(os.path.join(gt_dir, "*.png"))
        sd_files = glob.glob(os.path.join(sd_dir, "*.png"))
        vm_files = glob.glob(os.path.join(vm_dir, "*.png"))

        rgb_map = {_stem(p): p for p in rgb_files}
        gt_map = {_stem(p): p for p in gt_files}
        sd_map = {_stem(p): p for p in sd_files}
        vm_map = {_stem(p): p for p in vm_files}

        common = sorted(set(rgb_map.keys()) & set(gt_map.keys()) & set(sd_map.keys()) & set(vm_map.keys()))
        if len(common) == 0:
            if verbose:
                print(f"[warn] scene={sc} has 0 matched stems")
            continue

        if per_scene_cap > 0 and len(common) > per_scene_cap:
            idxs = np.linspace(0, len(common) - 1, per_scene_cap).round().astype(int).tolist()
            common_sel = [common[i] for i in idxs]
        else:
            common_sel = common

        for s in common_sel:
            pairs.append((rgb_map[s], sd_map[s], gt_map[s], vm_map[s]))

    if max_total > 0 and len(pairs) > max_total:
        rng = random.Random(int(seed))
        rng.shuffle(pairs)
        pairs = pairs[:max_total]

    if len(pairs) == 0:
        raise RuntimeError("No matched (rgb, sparse, gt, validity) pairs found.")

    if verbose:
        print(f"[scan] matched pairs: {len(pairs)}")
        rp, sp, gp, vp = pairs[0]
        print("[scan] sample pair:")
        print(" rgb:", rp)
        print(" sd :", sp)
        print(" gt :", gp)
        print(" vm :", vp)

    return pairs


# -------------------------
# VOID official split (train/test txt lists)
# -------------------------
def _read_list_txt(txt_path: str) -> List[str]:
    """Read a txt list file, ignoring empty lines and comments (#...)."""
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Cannot find txt list: {txt_path}")
    out: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    if len(out) == 0:
        raise RuntimeError(f"Empty txt list: {txt_path}")
    return out


def _candidate_bases(void_density_dir: str) -> List[str]:
    """Candidate base directories to resolve relative paths from official txt lists."""
    void_density_dir = os.path.normpath(void_density_dir)
    parent = os.path.dirname(void_density_dir)
    grand = os.path.dirname(parent)
    bases = [void_density_dir, parent, grand]
    seen = set()
    uniq = []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def _resolve_any_existing(void_density_dir: str, p: str) -> str:
    """Resolve a path line from official txt into an existing absolute path.

    VOID official txt sometimes contains:
      - data/... (relative to void_1500)
      - void_1500/data/... (relative to void_release root)
      - ./data/... (with './' prefix)

    This resolver tries multiple base dirs (density_dir, parent, grandparent) and
    also tries stripping the leading density dir name if present.
    """
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if os.path.isabs(p):
        return os.path.normpath(p)

    density_name = os.path.basename(os.path.normpath(void_density_dir)).replace("\\", "/")
    bases = _candidate_bases(void_density_dir)

    cand: List[str] = []

    def add_candidates(rel: str):
        for b in bases:
            cand.append(os.path.normpath(os.path.join(b, rel)))

    add_candidates(p)

    if p.startswith(density_name + "/"):
        stripped = p[len(density_name) + 1 :]
        add_candidates(stripped)

    # Deduplicate and return the first existing
    for c in cand:
        if os.path.isfile(c):
            return c

    # If none exists, return the first candidate for debugging
    return cand[0] if cand else os.path.normpath(os.path.join(void_density_dir, p))


def load_void_pairs_from_official_txt(void_density_dir: str, split: str, debug_show: int = 0) -> List[Tuple[str, str, str, str]]:
    """Load (rgb, sparse_depth, gt, validity_map) pairs from VOID official txt lists.

    Expected files under void_density_dir:
      - <split>_image.txt
      - <split>_sparse_depth.txt
      - <split>_validity_map.txt
      - <split>_ground_truth.txt
    """
    split = split.lower().strip()
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got: {split}")

    p_img = os.path.join(void_density_dir, f"{split}_image.txt")
    p_sd  = os.path.join(void_density_dir, f"{split}_sparse_depth.txt")
    p_vm  = os.path.join(void_density_dir, f"{split}_validity_map.txt")
    p_gt  = os.path.join(void_density_dir, f"{split}_ground_truth.txt")

    img_list = _read_list_txt(p_img)
    sd_list  = _read_list_txt(p_sd)
    vm_list  = _read_list_txt(p_vm)
    gt_list  = _read_list_txt(p_gt)

    n = len(img_list)
    if not (len(sd_list) == n and len(vm_list) == n and len(gt_list) == n):
        raise RuntimeError(
            "Official txt lists have different lengths: "
            f"image={len(img_list)}, sparse_depth={len(sd_list)}, validity_map={len(vm_list)}, ground_truth={len(gt_list)}"
        )

    if debug_show and debug_show > 0:
        print(f"[official:{split}] txt example lines (raw -> resolved -> exists):")
        for i in range(min(int(debug_show), n)):
            r_rgb = _resolve_any_existing(void_density_dir, img_list[i])
            r_sd  = _resolve_any_existing(void_density_dir, sd_list[i])
            r_gt  = _resolve_any_existing(void_density_dir, gt_list[i])
            r_vm  = _resolve_any_existing(void_density_dir, vm_list[i])
            print(f"  #{i}:")
            print(f"    raw rgb: {img_list[i]}")
            print(f"    res rgb: {r_rgb} | {os.path.isfile(r_rgb)}")
            print(f"    res sd : {r_sd} | {os.path.isfile(r_sd)}")
            print(f"    res gt : {r_gt} | {os.path.isfile(r_gt)}")
            print(f"    res vm : {r_vm} | {os.path.isfile(r_vm)}")

    pairs: List[Tuple[str, str, str, str]] = []
    missing = 0
    for i in range(n):
        rgb = _resolve_any_existing(void_density_dir, img_list[i])
        sd  = _resolve_any_existing(void_density_dir, sd_list[i])
        gt  = _resolve_any_existing(void_density_dir, gt_list[i])
        vm  = _resolve_any_existing(void_density_dir, vm_list[i])
        if not (os.path.isfile(rgb) and os.path.isfile(sd) and os.path.isfile(gt) and os.path.isfile(vm)):
            missing += 1
            continue
        pairs.append((rgb, sd, gt, vm))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No valid pairs after resolving txt lists (all missing?). "
            f"Check that --root points to the correct VOID density directory (e.g. .../void_1500). "
            f"Resolved base candidates: {_candidate_bases(void_density_dir)}"
        )

    if missing > 0:
        print(f"[warn] {missing} / {n} lines had missing files and were skipped.")

    print(f"[official] split={split} pairs={len(pairs)} (from {n} lines)")
    print("[official] sample pair:")
    rp, sp, gp, vp = pairs[0]
    print(" rgb:", rp)
    print(" sd :", sp)
    print(" gt :", gp)
    print(" vm :", vp)
    return pairs


def infer_void_density_dir(root: str, density: Optional[str] = None) -> str:
    """Infer the VOID density directory.

    Accepts:
      - root points directly to void_1500 (or void_500/void_150)
      - root points to void_release root, and density specifies which subset
    """
    root = os.path.normpath(root)
    cands: List[str] = []

    if density:
        cands.append(os.path.join(root, str(density)))

    # root itself
    cands.append(root)

    # common density folders
    for d in ["void_1500", "void_500", "void_150"]:
        cands.append(os.path.join(root, d))

    # de-dup
    seen = set()
    uniq = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    # prefer dirs that contain either official txts or data/
    for c in uniq:
        if not os.path.isdir(c):
            continue
        if os.path.isfile(os.path.join(c, "train_image.txt")) or os.path.isdir(os.path.join(c, "data")):
            return c

    raise FileNotFoundError(
        f"Cannot infer VOID density dir from root={root}. "
        f"Please pass --root as the density dir (e.g. .../void_1500) or pass --density void_1500 when root is void_release."
    )


def cap_pairs_by_scene_and_total(
    pairs: List[Tuple[str, str, str, str]],
    max_scenes: int = 0,
    per_scene_cap: int = 0,
    max_total: int = 0,
    seed: int = 42,
) -> List[Tuple[str, str, str, str]]:
    """Apply optional caps consistently for both scan and official modes."""
    out = pairs

    if (max_scenes and max_scenes > 0) or (per_scene_cap and per_scene_cap > 0):
        scene2pairs: Dict[str, List[Tuple[str, str, str, str]]] = {}
        for p in out:
            sid = _extract_scene_id(p[0])
            scene2pairs.setdefault(sid, []).append(p)

        scenes = sorted(scene2pairs.keys())
        if max_scenes and max_scenes > 0:
            scenes = scenes[: int(max_scenes)]

        capped: List[Tuple[str, str, str, str]] = []
        for sid in scenes:
            plist = sorted(scene2pairs[sid], key=lambda x: x[0])
            if per_scene_cap and per_scene_cap > 0 and len(plist) > int(per_scene_cap):
                idxs = np.linspace(0, len(plist) - 1, int(per_scene_cap)).round().astype(int).tolist()
                plist = [plist[i] for i in idxs]
            capped.extend(plist)
        out = capped

    if max_total and max_total > 0 and len(out) > int(max_total):
        rng = random.Random(int(seed))
        out = list(out)
        rng.shuffle(out)
        out = out[: int(max_total)]

    return out


# -------------------------
# Dataset (meters) + teacher mask for distillation
# -------------------------
class VoidDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str, str, str]],
        img_size=320,
        depth_scale=256.0,
        augment=True,
        # distillation
        teacher_enable: bool = False,
        teacher_subdir: str = "small_object_mask",
        teacher_ext: str = ".png",
        teacher_missing_as_zero: bool = True,
        # aug params
        aug_flip_p: float = 0.5,
        aug_color_p: float = 0.3,
        aug_rot_p: float = 0.2,
        aug_rot_deg: float = 10.0,
        # sparse-to-dense hint
        use_sd_fill: bool = True,
        # --- multi-frame fusion (cheap, no flow) ---
        temporal_radius: int = 0,
        temporal_mode: str = "union",   # union|avg (union is default/safest)
        # --- ROI crop to oversample small objects (uses teacher mask if available) ---
        roi_crop_p: float = 0.0,
        roi_crop_scale_min: float = 0.55,
        roi_crop_scale_max: float = 0.95,
        roi_crop_thr: float = 0.5,
    ):
        self.pairs = pairs
        self.img_size = img_size  # int (square) or (H,W)
        if isinstance(self.img_size, (tuple, list)):
            self.img_h, self.img_w = int(self.img_size[0]), int(self.img_size[1])
        else:
            self.img_h = self.img_w = int(self.img_size)
        self.depth_scale = float(depth_scale)
        self.augment = bool(augment)

        self.teacher_enable = bool(teacher_enable)
        self.teacher_subdir = str(teacher_subdir)
        self.teacher_ext = str(teacher_ext)
        self.teacher_missing_as_zero = bool(teacher_missing_as_zero)

        self.aug_flip_p = float(aug_flip_p)
        self.aug_color_p = float(aug_color_p)
        self.aug_rot_p = float(aug_rot_p)
        self.aug_rot_deg = float(aug_rot_deg)

        self.use_sd_fill = bool(use_sd_fill)

        # temporal fusion
        self.temporal_radius = max(0, int(temporal_radius))
        self.temporal_mode = str(temporal_mode).lower().strip()
        self._scene2idxs = None
        self._idx2scene = None
        self._idx2pos = None
        if self.temporal_radius > 0:
            self._build_temporal_index()

        # ROI crop
        self.roi_crop_p = float(roi_crop_p)
        self.roi_crop_scale_min = float(roi_crop_scale_min)
        self.roi_crop_scale_max = float(roi_crop_scale_max)
        self.roi_crop_thr = float(roi_crop_thr)

    def __len__(self):
        return len(self.pairs)

    def _build_temporal_index(self):
        """Build (scene -> ordered indices) for cheap neighbor lookup."""
        n = len(self.pairs)
        self._scene2idxs = {}
        self._idx2scene = ["" for _ in range(n)]
        self._idx2pos = [0 for _ in range(n)]
        for i, (rp, _, _, _) in enumerate(self.pairs):
            sid = _extract_scene_id(rp)
            self._scene2idxs.setdefault(sid, []).append(i)

        for sid, idxs in list(self._scene2idxs.items()):
            # Sort by rgb path string (works for zero-padded frame ids commonly used in VOID)
            idxs_sorted = sorted(idxs, key=lambda j: self.pairs[j][0].replace("\\", "/"))
            self._scene2idxs[sid] = idxs_sorted
            for pos, j in enumerate(idxs_sorted):
                self._idx2scene[j] = sid
                self._idx2pos[j] = pos

    def _temporal_neighbor_indices(self, idx: int) -> List[int]:
        if self.temporal_radius <= 0 or self._scene2idxs is None:
            return []
        sid = self._idx2scene[idx]
        idxs = self._scene2idxs.get(sid, [])
        if not idxs:
            return []
        pos = int(self._idx2pos[idx])
        nbrs = []
        for off in range(1, self.temporal_radius + 1):
            if pos - off >= 0:
                nbrs.append(idxs[pos - off])
            if pos + off < len(idxs):
                nbrs.append(idxs[pos + off])
        return nbrs

    def _teacher_path(self, rgb_path: str) -> str:
        # replace /image/ with /{teacher_subdir}/ and keep stem
        p = rgb_path.replace("\\", "/")
        if "/image/" in p:
            base = p.replace("/image/", f"/{self.teacher_subdir}/")
        elif "/rgb/" in p:
            base = p.replace("/rgb/", f"/{self.teacher_subdir}/")
        else:
            # fallback: sibling folder
            base = os.path.join(os.path.dirname(rgb_path), "..", self.teacher_subdir, os.path.basename(rgb_path))
        stem = _stem(base)
        return os.path.join(os.path.dirname(base), stem + self.teacher_ext)

    def _apply_roi_crop(self, rgb, sd, gt, vm, mteach, neigh_sd_list, neigh_vm_list):
        """Zoom-in crop around teacher mask (or sparse-valid pixels) to oversample small objects."""
        if (not self.augment) or (self.roi_crop_p <= 0) or (random.random() >= self.roi_crop_p):
            return rgb, sd, gt, vm, mteach, neigh_sd_list, neigh_vm_list

        H, W = rgb.shape[:2]
        smin = max(0.2, min(self.roi_crop_scale_min, 1.0))
        smax = max(smin, min(self.roi_crop_scale_max, 1.0))
        s = random.uniform(smin, smax)
        ch = int(round(H * s))
        cw = int(round(W * s))
        ch = max(32, min(H, ch))
        cw = max(32, min(W, cw))

        cy, cx = None, None
        if mteach is not None:
            ys, xs = np.where(mteach > self.roi_crop_thr)
            if ys.size > 0:
                k = random.randrange(int(ys.size))
                cy, cx = int(ys[k]), int(xs[k])

        if cy is None:
            ys, xs = np.where(vm > 0.5)
            if ys.size > 0 and random.random() < 0.5:
                k = random.randrange(int(ys.size))
                cy, cx = int(ys[k]), int(xs[k])
            else:
                cy = random.randrange(H)
                cx = random.randrange(W)

        y1 = int(np.clip(cy - ch // 2, 0, H - ch))
        x1 = int(np.clip(cx - cw // 2, 0, W - cw))
        y2 = y1 + ch
        x2 = x1 + cw

        rgb_c = rgb[y1:y2, x1:x2, :]
        sd_c = sd[y1:y2, x1:x2]
        gt_c = gt[y1:y2, x1:x2]
        vm_c = vm[y1:y2, x1:x2]
        mt_c = mteach[y1:y2, x1:x2] if mteach is not None else None

        neigh_sd_c = [n[y1:y2, x1:x2] for n in neigh_sd_list]
        neigh_vm_c = [n[y1:y2, x1:x2] for n in neigh_vm_list]

        S = (H, W)
        rgb = safe_resize(rgb_c, S, cv2.INTER_LINEAR)
        sd = safe_resize(sd_c, S, cv2.INTER_NEAREST)
        gt = safe_resize(gt_c, S, cv2.INTER_NEAREST)
        vm = safe_resize(vm_c, S, cv2.INTER_NEAREST)
        if mt_c is not None:
            mteach = safe_resize(mt_c, S, cv2.INTER_NEAREST)
        else:
            mteach = None

        neigh_sd_list = [safe_resize(n, S, cv2.INTER_NEAREST) for n in neigh_sd_c]
        neigh_vm_list = [safe_resize(n, S, cv2.INTER_NEAREST) for n in neigh_vm_c]
        return rgb, sd, gt, vm, mteach, neigh_sd_list, neigh_vm_list

    def __getitem__(self, idx):
        rp, sp, gp, vp = self.pairs[idx]
        rgb = read_rgb(rp)

        sd_u16 = read_u16(sp).astype(np.float32)
        gt_u16 = read_u16(gp).astype(np.float32)
        vm_u16 = read_u16(vp).astype(np.float32)

        sd = sd_u16 / self.depth_scale
        gt = gt_u16 / self.depth_scale
        vm = ((vm_u16 > 0) & (gt_u16 < 65000)).astype(np.float32)

        # teacher mask
        mteach = None
        if self.teacher_enable:
            tp = self._teacher_path(rp)
            mteach = read_mask01(tp)
            # if missing, we will fill with zeros after resize

        # ---- temporal neighbors (sparse depth + validity only) ----
        neigh_sd_list = []
        neigh_vm_list = []
        if self.temporal_radius > 0:
            nbrs = self._temporal_neighbor_indices(idx)
            for j in nbrs:
                _, spj, _, vpj = self.pairs[j]
                sdj = read_u16(spj).astype(np.float32) / self.depth_scale
                vmj = (read_u16(vpj).astype(np.float32) > 0).astype(np.float32)
                neigh_sd_list.append(sdj)
                neigh_vm_list.append(vmj)

        # resize all to training size first
        S = (self.img_h, self.img_w)
        rgb = safe_resize(rgb, S, cv2.INTER_LINEAR)
        sd = safe_resize(sd, S, cv2.INTER_NEAREST)
        gt = safe_resize(gt, S, cv2.INTER_NEAREST)
        vm = safe_resize(vm, S, cv2.INTER_NEAREST)

        if self.teacher_enable:
            if mteach is None:
                mteach = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            else:
                mteach = safe_resize(mteach, S, cv2.INTER_NEAREST)
                mteach = np.clip(mteach, 0.0, 1.0).astype(np.float32)

        if self.temporal_radius > 0 and len(neigh_sd_list) > 0:
            neigh_sd_list = [safe_resize(n, S, cv2.INTER_NEAREST) for n in neigh_sd_list]
            neigh_vm_list = [safe_resize(n, S, cv2.INTER_NEAREST) for n in neigh_vm_list]

        # ROI crop (teacher-guided)
        if self.augment and self.roi_crop_p > 0:
            rgb, sd, gt, vm, mteach, neigh_sd_list, neigh_vm_list = self._apply_roi_crop(
                rgb, sd, gt, vm, mteach, neigh_sd_list, neigh_vm_list
            )

        # ---- augmentations (shared across current + neighbors) ----
        if self.augment:
            # flip (H flip)
            if random.random() < self.aug_flip_p:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                sd = np.ascontiguousarray(sd[:, ::-1])
                gt = np.ascontiguousarray(gt[:, ::-1])
                vm = np.ascontiguousarray(vm[:, ::-1])
                if mteach is not None:
                    mteach = np.ascontiguousarray(mteach[:, ::-1])
                if len(neigh_sd_list) > 0:
                    neigh_sd_list = [np.ascontiguousarray(n[:, ::-1]) for n in neigh_sd_list]
                    neigh_vm_list = [np.ascontiguousarray(n[:, ::-1]) for n in neigh_vm_list]

            # color jitter (RGB only)
            if random.random() < self.aug_color_p:
                rgbf = rgb.astype(np.float32) / 255.0
                c = 1.0 + (random.random() * 2 - 1) * 0.2   # +/-0.2
                b = (random.random() * 2 - 1) * 0.2         # +/-0.2
                rgbf = np.clip(rgbf * c + b, 0, 1)
                rgb = (rgbf * 255.0).astype(np.uint8)

            # rotate (aligned, including temporal neighbors)
            if random.random() < self.aug_rot_p:
                ang = (random.random() * 2 - 1) * self.aug_rot_deg
                H, W = rgb.shape[:2]
                M = cv2.getRotationMatrix2D((W * 0.5, H * 0.5), ang, 1.0)
                rgb = cv2.warpAffine(
                    rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                sd = cv2.warpAffine(
                    sd, M, (W, H), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                gt = cv2.warpAffine(
                    gt, M, (W, H), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                vm = cv2.warpAffine(
                    vm, M, (W, H), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                if mteach is not None:
                    mteach = cv2.warpAffine(
                        mteach, M, (W, H), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                if len(neigh_sd_list) > 0:
                    neigh_sd_list = [
                        cv2.warpAffine(n, M, (W, H), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        for n in neigh_sd_list
                    ]
                    neigh_vm_list = [
                        cv2.warpAffine(n, M, (W, H), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        for n in neigh_vm_list
                    ]

        # ---- sparse-to-dense hint (optionally multi-frame union) ----
        sd_fill = None
        if self.use_sd_fill:
            if self.temporal_radius > 0 and len(neigh_sd_list) > 0:
                # union/avg fusion only affects the *hint* channel; we keep (sd,vm) as current-frame measurement
                sd_u = sd.copy()
                vm_u = vm.copy()
                if self.temporal_mode == "avg":
                    # average where any neighbor has measurement (still no alignment; use with caution)
                    acc = sd_u * vm_u
                    cnt = vm_u
                    for sd_n, vm_n in zip(neigh_sd_list, neigh_vm_list):
                        acc = acc + sd_n * vm_n
                        cnt = cnt + vm_n
                    cnt = np.clip(cnt, 0.0, 10.0)
                    sd_u = np.where(cnt > 0.5, acc / (cnt + 1e-6), sd_u)
                    vm_u = (cnt > 0.5).astype(np.float32)
                else:
                    # default: union fill missing pixels only (safest)
                    for sd_n, vm_n in zip(neigh_sd_list, neigh_vm_list):
                        m = (vm_u < 0.5) & (vm_n > 0.5)
                        sd_u[m] = sd_n[m]
                        vm_u[m] = 1.0
                sd_fill = fill_sparse_depth_nearest(sd_u, vm_u)
            else:
                sd_fill = fill_sparse_depth_nearest(sd, vm)

        gt_valid = (gt > 0).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        sd_t = torch.from_numpy(sd).float().unsqueeze(0)
        vm_t = torch.from_numpy(vm).float().unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        gtv_t = torch.from_numpy(gt_valid).float().unsqueeze(0)

        if sd_fill is not None:
            sd_fill_t = torch.from_numpy(sd_fill).float().unsqueeze(0)
            x = torch.cat([rgb_t, sd_t, vm_t, sd_fill_t], dim=0)
        else:
            x = torch.cat([rgb_t, sd_t, vm_t], dim=0)

        if self.teacher_enable:
            mt_t = torch.from_numpy(mteach).float().unsqueeze(0)
            return x, gt_t, gtv_t, mt_t
        return x, gt_t, gtv_t


# -------------------------
# Building blocksblocks
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class DWConvBNAct(nn.Module):
    """Depthwise-separable conv: cheap local adapter."""
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class PatchEmbed(nn.Module):
    def __init__(self, patch=8, in_ch=32, dim=32):
        super().__init__()
        self.patch = max(1, int(patch))
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=self.patch, stride=self.patch)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PatchUnembed(nn.Module):
    def __init__(self, patch=8, dim=32):
        super().__init__()
        self.patch = max(1, int(patch))
        self.dim = dim

    def forward(self, tokens, H, W):
        B, N, D = tokens.shape
        Hp, Wp = H // self.patch, W // self.patch
        x = tokens.transpose(1, 2).contiguous().view(B, D, Hp, Wp)
        x = F.interpolate(x, size=(H, W), mode="nearest")
        return x


# -------------------------
# SOFA
# -------------------------
class SofaCrossAttention(nn.Module):
    def __init__(self, dim=32, heads=4, patch=8, bias_scale=2.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.patch = max(1, int(patch))
        self.bias_scale = float(bias_scale)

        self.pe_q = PatchEmbed(self.patch, dim, dim)
        self.pe_kv = PatchEmbed(self.patch, dim, dim)
        self.un = PatchUnembed(self.patch, dim)

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, f_q: torch.Tensor, f_kv: torch.Tensor, mask_so: torch.Tensor, return_stats: bool = False, attn_thresh: float = 0.5):
        """Cross attention from query features f_q to key/value features f_kv,
        with a soft attention bias derived from mask_so.

        If return_stats=True, also returns a dict containing lightweight foveation statistics
        (e.g., attention mass on foveal tokens and attention entropy). This is intended for
        paper-ready logging during validation, and should generally be disabled during training.
        """
        B, C, H, W = f_q.shape
        tq = self.pe_q(f_q)     # (B, Nq, C)
        tkv = self.pe_kv(f_kv)  # (B, Nk, C)

        # Downsample mask to token grid and flatten to (B, Nk)
        mk = F.interpolate(mask_so, size=(H // self.patch, W // self.patch), mode="nearest")
        mk = mk.flatten(2).transpose(1, 2).squeeze(-1)  # (B, Nk)

        # Attention bias: suppress non-foveal tokens for KV
        bias = -self.bias_scale * (1.0 - mk).clamp(0, 1)          # (B, Nk)
        bias = bias.unsqueeze(1).repeat(1, tq.shape[1], 1)        # (B, Nq, Nk)
        bias = bias.repeat_interleave(self.heads, dim=0)          # (B*heads, Nq, Nk)

        stats = None
        if return_stats:
            # Note: requesting attention weights increases memory; keep validation batch small (we use batch=1).
            attn_out, attn_w = self.attn(tq, tkv, tkv, attn_mask=bias, need_weights=True, average_attn_weights=False)
            # attn_w: (B, heads, Nq, Nk)
            # Compute foveal token mask
            fmask = (mk > float(attn_thresh)).to(attn_w.dtype)  # (B, Nk)
            fmask = fmask.unsqueeze(1).unsqueeze(2)            # (B,1,1,Nk)
            # Attention mass on foveal tokens
            mass_f = (attn_w * fmask).sum(dim=-1)              # (B,heads,Nq)
            mass_all = attn_w.sum(dim=-1).clamp(min=1e-12)
            mass_ratio = (mass_f / mass_all).mean().detach()
            # Attention entropy (per query)
            p = (attn_w / mass_all.unsqueeze(-1)).clamp(min=1e-12)
            ent = (-(p * p.log()).sum(dim=-1)).mean().detach()
            # Foveal token area ratio on token grid
            area_ratio = fmask.mean().detach()
            stats = {
                "sofa_attn_mass_fovea": float(mass_ratio.item()),
                "sofa_attn_entropy": float(ent.item()),
                "sofa_fovea_token_area": float(area_ratio.item()),
            }
        else:
            attn_out, _ = self.attn(tq, tkv, tkv, attn_mask=bias, need_weights=False)

        tq = self.norm1(tq + attn_out)
        tq = self.norm2(tq + self.ff(tq))
        out = self.un(tq, H, W)
        return (out, stats) if return_stats else out

class WindowAttentionBlock(nn.Module):
    """Window self-attention block (Swin-like) with an extra depthwise conv mixing.
    The depthwise conv is a cheap way to communicate across window boundaries,
    improving detail recovery without a big FLOPs increase.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        ws: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        mix_dwconv: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.ws = int(ws)

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, heads, batch_first=True, dropout=drop)

        self.norm2 = nn.LayerNorm(self.dim)
        hid = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, self.dim),
        )

        # cheap spatial mixing across windows
        self.mix_dwconv = bool(mix_dwconv)
        if self.mix_dwconv:
            self.dw = nn.Conv2d(self.dim, self.dim, 3, 1, 1, groups=self.dim, bias=False)
            self.dw_bn = nn.BatchNorm2d(self.dim)
            self.dw_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, meta = _window_partition(x, self.ws)
        w = self.norm1(windows)
        out, _ = self.attn(w, w, w, need_weights=False)
        windows = windows + out
        windows = windows + self.mlp(self.norm2(windows))

        x = _window_reverse(windows, self.ws, meta)

        if self.mix_dwconv:
            x = x + self.dw_act(self.dw_bn(self.dw(x)))
        return x


# -------------------------
# YOLO -> small-object mask (optional, online only; not needed for distill)
# -------------------------
class YoloMaskGenerator(nn.Module):
    def __init__(
        self,
        ckpt: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        small_area_max: int = 48 * 48,
        device: str = "cuda",
    ):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO. Install with: pip install ultralytics")
        self.det = YOLO(ckpt)
        self.conf = float(conf)
        self.iou = float(iou)
        self.small_area_max = int(small_area_max)
        self.device = device

    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb01.shape
        masks = []
        for b in range(B):
            img = (rgb01[b].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            res = self.det.predict(img, conf=self.conf, iou=self.iou, verbose=False, device=self.device)
            r0 = res[0]
            m = np.zeros((H, W), dtype=np.float32)
            if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
                boxes = r0.boxes.xyxy.detach().cpu().numpy()
                for (x1, y1, x2, y2) in boxes:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, min(W - 1, x1))
                    x2 = max(0, min(W, x2))
                    y1 = max(0, min(H - 1, y1))
                    y2 = max(0, min(H, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    if area <= self.small_area_max:
                        m[y1:y2, x1:x2] = 1.0
            masks.append(torch.from_numpy(m).unsqueeze(0))
        return torch.stack(masks, dim=0).to(rgb01.device)


# -------------------------
# Transformer backbone (window-attention U-Net)
# -------------------------
def _window_partition(x: torch.Tensor, ws: int):
    B, C, H, W = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nh, nw, ws, ws, C
    windows = x.view(B * (Hp // ws) * (Wp // ws), ws * ws, C)
    return windows, (H, W, Hp, Wp, pad_h, pad_w)


def _window_reverse(windows: torch.Tensor, ws: int, meta):
    H, W, Hp, Wp, pad_h, pad_w = meta
    nW = (Hp // ws) * (Wp // ws)
    B = windows.shape[0] // nW
    C = windows.shape[-1]
    x = windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
    if pad_h or pad_w:
        x = x[:, :, :H, :W]
    return x


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, ws: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.ws = int(ws)
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(self.dim)
        hid = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(self.dim, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, meta = _window_partition(x, self.ws)
        w = self.norm1(windows)
        out, _ = self.attn(w, w, w, need_weights=False)
        windows = windows + out
        windows = windows + self.mlp(self.norm2(windows))
        return _window_reverse(windows, self.ws, meta)


class TransformerUNetBackbone(nn.Module):
    def __init__(self, base: int = 32, heads: int = 4, ws: int = 8, depth: int = 2):
        super().__init__()
        b = int(base)

        def stage(ch):
            return nn.Sequential(*[WindowAttentionBlock(ch, heads=heads, ws=ws) for _ in range(depth)])

        self.enc0 = stage(b)
        self.down1 = nn.Conv2d(b, b * 2, 3, 2, 1)
        self.enc1 = stage(b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 4, 3, 2, 1)
        self.enc2 = stage(b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 8, 3, 2, 1)
        self.bott = stage(b * 8)

        self.up2 = nn.Sequential(nn.Conv2d(b * 8 + b * 4, b * 4, 1), stage(b * 4))
        self.up1 = nn.Sequential(nn.Conv2d(b * 4 + b * 2, b * 2, 1), stage(b * 2))
        self.up0 = nn.Sequential(nn.Conv2d(b * 2 + b, b, 1), stage(b))

    def forward(self, x0: torch.Tensor):
        s0 = self.enc0(x0)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        s3 = self.bott(self.down3(s2))

        y = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, s2], dim=1))
        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, s1], dim=1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up0(torch.cat([y, s0], dim=1))
        return y


# -------------------------
# PPE
# -------------------------
class PPE(nn.Module):
    def __init__(self, ch: int, learn_kmap: bool = True):
        super().__init__()
        self.learn_kmap = bool(learn_kmap)
        self.k_raw = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.proj = nn.Sequential(nn.Conv2d(4, ch, 1), nn.SiLU(inplace=True), nn.Conv2d(ch, ch, 1))
        if self.learn_kmap:
            self.kmap = nn.Sequential(ConvBNAct(ch, ch, 3, 1, 1), nn.Conv2d(ch, 1, 1))

    def forward(self, feat: torch.Tensor, z_m: torch.Tensor):
        B, C, H, W = feat.shape
        z_m = z_m.clamp_min(1e-3)
        z2 = z_m * z_m

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        k = F.softplus(self.k_raw)
        z2_scaled = k * z2
        prior = torch.cat([z_m, z2_scaled, pos], dim=1)
        bias = self.proj(prior)

        if self.learn_kmap:
            kmap = self.kmap(feat).tanh()
            bias = bias + 0.1 * kmap * bias
        return feat + bias


# -------------------------
# MIM
# -------------------------
def rgb_to_value_saturation(rgb: torch.Tensor):
    mx, _ = rgb.max(dim=1, keepdim=True)
    mn, _ = rgb.min(dim=1, keepdim=True)
    v = mx
    s = (mx - mn) / (mx + 1e-6)
    return v, s


class MIM(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Sequential(ConvBNAct(ch + 2, ch, 3, 1, 1), nn.Conv2d(ch, 1, 1))
        self.bias = nn.Sequential(ConvBNAct(ch + 2, ch, 3, 1, 1), ConvBNAct(ch, ch, 3, 1, 1))
        self.sig_lift = nn.Sequential(ConvBNAct(ch + 2, ch, 3, 1, 1), nn.Conv2d(ch, 1, 1))

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor):
        dx = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).mean(1, keepdim=True)
        dy = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).mean(1, keepdim=True)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = (dx + dy).clamp(0, 1)

        v, s = rgb_to_value_saturation(rgb)
        mpi = (v > 0.8).float() * (s < 0.25).float()

        # Two-channel conditioning: (edge, mpi)
        cond = torch.cat([edge, mpi], dim=1)  # [B,2,H,W]
        x = torch.cat([feat, cond], dim=1)   # [B,ch+2,H,W]

        gate = torch.sigmoid(self.gate(x))
        bias = self.bias(x)
        sig_lift = self.sig_lift(x)

        # Feature injection; keep identity when gate ~ 0
        out = feat + gate * bias
        return out, mpi, sig_lift

class UACSPNRefiner(nn.Module):
    """Uncertainty-Aware CSPN-lite refinement (8-neighbor + center, softmax normalized).

    Why this exists:
    - LiteLearnedPropRefiner (your BP module) is already good for global smoothing.
    - But small objects / thin structures often need stronger, edge-aware local propagation.
    - This module adds a very light CSPN-like iterative refinement:
        * per-pixel normalized affinities (softmax)
        * optional edge barrier so propagation won't cross strong RGB edges
        * optional ROI boost via mask_so (stronger propagation inside ROI)

    It is designed to be lightweight enough for training on a modern CUDA GPU.
    """
    def __init__(self, iters: int = 3, guide_ch: int = 32, hidden: int = 16, use_diag: bool = True):
        super().__init__()
        self.iters = max(0, int(iters))
        self.use_diag = bool(use_diag)
        self.K = 9 if self.use_diag else 5  # center + 8 or center + 4

        in_ch = int(guide_ch) + 1 + 1 + 1 + 2  # guide + mask_so + logvar + vm + (edge_x, edge_y)
        hid = int(hidden)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, self.K + 1, 1, 1, 0),  # weights logits + gate logits
        )

        # init: center weight slightly larger; gate small
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[0] = 1.0   # center weight bias
            self.net[-1].bias[-1] = -2.0 # gate bias => small updates at start

        # edge barrier strength (learnable)
        self.alpha_raw = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

    def _edges(self, rgb: torch.Tensor):
        gray = rgb.mean(1, keepdim=True)
        dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        dy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx.clamp(0, 1), dy.clamp(0, 1)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        rgb: torch.Tensor,
        guide: torch.Tensor,
        mask_so: Optional[torch.Tensor] = None,
        vm: Optional[torch.Tensor] = None,
        anchor_mu: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.iters <= 0:
            return mu

        B, _, H, W = mu.shape
        if mask_so is None:
            mask_so = torch.zeros((B, 1, H, W), device=mu.device, dtype=mu.dtype)
        if vm is None:
            vm = torch.zeros((B, 1, H, W), device=mu.device, dtype=mu.dtype)
        if obs is None:
            obs = (vm > 0.5)
        if anchor_mu is None:
            anchor_mu = mu.detach()

        # edge maps (for barrier)
        edge_x, edge_y = self._edges(rgb)

        # predict affinities + gate
        lv = logvar.clamp(-12.0, 8.0)
        inp = torch.cat([guide, mask_so, lv, vm, edge_x, edge_y], dim=1)
        out = self.net(inp)
        w_logits = out[:, : self.K]
        gate = torch.sigmoid(out[:, self.K : self.K + 1])

        w = F.softmax(w_logits, dim=1)

        # edge barrier: reduce cross-edge propagation
        alpha = F.softplus(self.alpha_raw) + 1e-6
        bar_lr = torch.exp(-alpha * edge_x)
        bar_ud = torch.exp(-alpha * edge_y)
        if self.K == 9:
            bar_diag = torch.exp(-alpha * 0.5 * (edge_x + edge_y))
            w0 = w[:, 0:1]  # center
            w1 = w[:, 1:2] * bar_ud  # up
            w2 = w[:, 2:3] * bar_ud  # down
            w3 = w[:, 3:4] * bar_lr  # left
            w4 = w[:, 4:5] * bar_lr  # right
            w5 = w[:, 5:6] * bar_diag  # ul
            w6 = w[:, 6:7] * bar_diag  # ur
            w7 = w[:, 7:8] * bar_diag  # dl
            w8 = w[:, 8:9] * bar_diag  # dr
            w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7, w8], dim=1)
        else:
            w0 = w[:, 0:1]
            w1 = w[:, 1:2] * bar_ud
            w2 = w[:, 2:3] * bar_ud
            w3 = w[:, 3:4] * bar_lr
            w4 = w[:, 4:5] * bar_lr
            w = torch.cat([w0, w1, w2, w3, w4], dim=1)

        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        # stronger propagation inside ROI
        gate_s = gate * (0.35 + 0.65 * mask_so)

        # iterative propagation (replicate padding to avoid wrap-around)
        for _ in range(self.iters):
            mu_pad = F.pad(mu, (1, 1, 1, 1), mode="replicate")
            mu_c = mu
            mu_up = mu_pad[:, :, 0:H, 1 : 1 + W]
            mu_dn = mu_pad[:, :, 2 : 2 + H, 1 : 1 + W]
            mu_lt = mu_pad[:, :, 1 : 1 + H, 0:W]
            mu_rt = mu_pad[:, :, 1 : 1 + H, 2 : 2 + W]

            if self.K == 9:
                mu_ul = mu_pad[:, :, 0:H, 0:W]
                mu_ur = mu_pad[:, :, 0:H, 2 : 2 + W]
                mu_dl = mu_pad[:, :, 2 : 2 + H, 0:W]
                mu_dr = mu_pad[:, :, 2 : 2 + H, 2 : 2 + W]
                stack = torch.cat([mu_c, mu_up, mu_dn, mu_lt, mu_rt, mu_ul, mu_ur, mu_dl, mu_dr], dim=1)
            else:
                stack = torch.cat([mu_c, mu_up, mu_dn, mu_lt, mu_rt], dim=1)

            mu_prop = (w * stack).sum(dim=1, keepdim=True)
            mu = mu + gate_s * (mu_prop - mu)

            # keep observed (fused) measurements stable
            mu = torch.where(obs, anchor_mu, mu)

        return mu
# -------------------------
# GBPN-like refiner
# -------------------------
class GaussianBPRefiner(nn.Module):
    """GBPN-inspired spatial refinement for (mu, logvar).

    Important stability fix:
    - Naively accumulating neighbor precisions can collapse variance (overconfidence),
      which explodes the Gaussian NLL and hurts depth training.
    - We introduce a (softplus) scaled neighbor precision weight `beta` and allow
      optionally keeping logvar unchanged (refine_var=False).
    """

    def __init__(
        self,
        iters: int = 3,
        beta_init: float = 0.2,
        refine_var: bool = False,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.iters = int(iters)
        self.alpha_raw = nn.Parameter(torch.tensor(6.0, dtype=torch.float32))
        # beta in (0,+inf) via softplus; initialize near beta_init
        self.beta_raw = nn.Parameter(torch.tensor(inv_softplus(max(float(beta_init), 1e-4)), dtype=torch.float32))
        self.refine_var = bool(refine_var)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, rgb: torch.Tensor):
        dx = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).mean(1, keepdim=True)
        dy = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).mean(1, keepdim=True)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = (dx + dy).clamp(0, 1)

        alpha = F.softplus(self.alpha_raw)
        w = torch.exp(-alpha * edge)

        beta = F.softplus(self.beta_raw)

        def shift(x, dy, dx):
            return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

        mu_i = mu
        logv_i = logvar.clamp(self.min_logvar, self.max_logvar)

        for _ in range(self.iters):
            var = torch.exp(logv_i).clamp(1e-6, 1e6)
            prec = 1.0 / var

            mu_up = shift(mu_i, -1, 0); prec_up = shift(prec, -1, 0); w_up = shift(w, -1, 0)
            mu_dn = shift(mu_i, 1, 0);  prec_dn = shift(prec, 1, 0);  w_dn = shift(w, 1, 0)
            mu_lt = shift(mu_i, 0, -1); prec_lt = shift(prec, 0, -1); w_lt = shift(w, 0, -1)
            mu_rt = shift(mu_i, 0, 1);  prec_rt = shift(prec, 0, 1);  w_rt = shift(w, 0, 1)

            prec_n = (w_up * prec_up + w_dn * prec_dn + w_lt * prec_lt + w_rt * prec_rt)
            mu_prec_n = (w_up * prec_up * mu_up + w_dn * prec_dn * mu_dn + w_lt * prec_lt * mu_lt + w_rt * prec_rt * mu_rt)

            # damp neighbor contribution to avoid overconfidence
            prec_new = prec + beta * prec_n
            mu_new = (prec * mu_i + beta * mu_prec_n) / (prec_new + 1e-6)
            mu_i = mu_new

            if self.refine_var:
                var_new = 1.0 / prec_new.clamp(1e-6, 1e6)
                logv_i = torch.log(var_new).clamp(self.min_logvar, self.max_logvar)

        return mu_i, logv_i



class LiteLearnedPropRefiner(nn.Module):
    """Learned 4-neighbor propagation with minimal overhead.

    Predict per-pixel directional weights (up/down/left/right) + a gate using a tiny 1x1 network.
    Then perform precision-weighted message passing updates (stable and fast).
    """
    def __init__(
        self,
        iters: int = 2,
        guide_ch: int = 32,
        hidden: int = 8,
        beta_init: float = 0.2,
        refine_var: bool = False,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
    ):
        super().__init__()
        self.iters = int(iters)
        self.refine_var = bool(refine_var)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        # edge barrier + neighbor precision scale (stable)
        self.alpha_raw = nn.Parameter(torch.tensor(6.0, dtype=torch.float32))
        self.beta_raw  = nn.Parameter(torch.tensor(inv_softplus(max(float(beta_init), 1e-4)), dtype=torch.float32))

        # input: guide + dx + dy + mask_so + logvar  => guide_ch + 4
        in_ch = int(guide_ch) + 4
        h = int(hidden)

        # super light predictor (1x1): 4 dir logits + 1 gate
        self.pred = nn.Sequential(
            nn.Conv2d(in_ch, h, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, 5, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.pred[-1].weight)
        nn.init.zeros_(self.pred[-1].bias)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        rgb: torch.Tensor,
        guide: torch.Tensor,
        mask_so: Optional[torch.Tensor] = None,
    ):
        # directional rgb diffs (cheap, no conv)
        dx = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).mean(1, keepdim=True)
        dy = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).mean(1, keepdim=True)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        if mask_so is None:
            mask_so = torch.zeros_like(mu)

        logv = logvar.clamp(self.min_logvar, self.max_logvar)

        inp = torch.cat([guide, dx, dy, mask_so, logv], dim=1)
        out = self.pred(inp)

        dir_logits = out[:, :4]              # (B,4,H,W)
        gate = torch.sigmoid(out[:, 4:5])    # (B,1,H,W)

        # normalize directional weights
        dir_w = F.softmax(dir_logits, dim=1)
        w_up = dir_w[:, 0:1]
        w_dn = dir_w[:, 1:2]
        w_lt = dir_w[:, 2:3]
        w_rt = dir_w[:, 3:4]

        # edge barrier (keep boundary respect)
        alpha = F.softplus(self.alpha_raw)
        w_lr = torch.exp(-alpha * dx)
        w_ud = torch.exp(-alpha * dy)
        w_up = w_up * w_ud
        w_dn = w_dn * w_ud
        w_lt = w_lt * w_lr
        w_rt = w_rt * w_lr

        # renormalize after applying barrier
        w_sum = (w_up + w_dn + w_lt + w_rt).clamp_min(1e-6)
        w_up = w_up / w_sum
        w_dn = w_dn / w_sum
        w_lt = w_lt / w_sum
        w_rt = w_rt / w_sum

        # ROI-aware strength: more propagation where mask_so is high
        gate = gate * (0.5 + 0.5 * mask_so)

        beta = F.softplus(self.beta_raw)

        def shift(x, dy, dx):
            return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

        mu_i = mu
        logv_i = logv

        if not self.refine_var:
            var = torch.exp(logv_i).clamp(1e-6, 1e6)
            prec = 1.0 / var

            prec_up = shift(prec, -1, 0)
            prec_dn = shift(prec,  1, 0)
            prec_lt = shift(prec,  0,-1)
            prec_rt = shift(prec,  0, 1)

            prec_n = (w_up * prec_up + w_dn * prec_dn + w_lt * prec_lt + w_rt * prec_rt)
            prec_new = prec + beta * gate * prec_n

            for _ in range(self.iters):
                mu_up = shift(mu_i, -1, 0)
                mu_dn = shift(mu_i,  1, 0)
                mu_lt = shift(mu_i,  0,-1)
                mu_rt = shift(mu_i,  0, 1)

                mu_prec_n = (
                    w_up * prec_up * mu_up +
                    w_dn * prec_dn * mu_dn +
                    w_lt * prec_lt * mu_lt +
                    w_rt * prec_rt * mu_rt
                )
                mu_i = (prec * mu_i + beta * gate * mu_prec_n) / (prec_new + 1e-6)

            return mu_i, logv_i

        # slower path: refine var too
        for _ in range(self.iters):
            var = torch.exp(logv_i).clamp(1e-6, 1e6)
            prec = 1.0 / var

            mu_up = shift(mu_i, -1, 0); prec_up = shift(prec, -1, 0)
            mu_dn = shift(mu_i,  1, 0); prec_dn = shift(prec,  1, 0)
            mu_lt = shift(mu_i,  0,-1); prec_lt = shift(prec,  0,-1)
            mu_rt = shift(mu_i,  0, 1); prec_rt = shift(prec,  0, 1)

            prec_n = (w_up * prec_up + w_dn * prec_dn + w_lt * prec_lt + w_rt * prec_rt)
            mu_prec_n = (
                w_up * prec_up * mu_up +
                w_dn * prec_dn * mu_dn +
                w_lt * prec_lt * mu_lt +
                w_rt * prec_rt * mu_rt
            )

            prec_new = prec + beta * gate * prec_n
            mu_i = (prec * mu_i + beta * gate * mu_prec_n) / (prec_new + 1e-6)

            var_new = 1.0 / prec_new.clamp(1e-6, 1e6)
            logv_i = torch.log(var_new).clamp(self.min_logvar, self.max_logvar)

        return mu_i, logv_i



# -------------------------
# BFS-Head (new small-object head)
# -------------------------
class BFSHead(nn.Module):
    def __init__(self, in_ch: int, base: int, p: float = 0.05, t: float = 0.1):
        super().__init__()
        self.p = float(p)  # target mass proportion (not used here, passed to loss)
        self.t = float(t)  # temperature

        b4 = base // 4
        # Scale branches (with dilation)
        self.r1 = ConvBNAct(in_ch, b4, 3, 1, 1, dilation=1)
        self.r2 = ConvBNAct(in_ch, b4, 3, 1, 2, dilation=2)  # effective RF=5
        self.r4 = ConvBNAct(in_ch, b4, 3, 1, 4, dilation=4)  # effective RF=9

        # Alpha proj for scale weights
        self.alpha_proj = nn.Conv2d(3 * b4, 3, 1)

        # Logit proj per scale
        self.logit_proj1 = nn.Conv2d(b4, 1, 1)
        self.logit_proj2 = nn.Conv2d(b4, 1, 1)
        self.logit_proj4 = nn.Conv2d(b4, 1, 1)

        # Tau (adaptive threshold)
        self.tau_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, b4),
            nn.ReLU(),
            nn.Linear(b4, 1),
        )

    def forward(self, F_in: torch.Tensor):
        r1 = self.r1(F_in)
        r2 = self.r2(F_in)
        r4 = self.r4(F_in)

        # Scale weights alpha
        cat_r = torch.cat([r1, r2, r4], dim=1)
        alpha = self.alpha_proj(cat_r).softmax(dim=1)  # (B,3,H,W)
        a1, a2, a4 = alpha.unbind(1)  # each (B,1,H,W) after unsqueeze if needed
        a1 = a1.unsqueeze(1)
        a2 = a2.unsqueeze(1)
        a4 = a4.unsqueeze(1)

        # Logits per scale
        l1 = self.logit_proj1(r1)
        l2 = self.logit_proj2(r2)
        l4 = self.logit_proj4(r4)

        logits = a1 * l1 + a2 * l2 + a4 * l4

        # Tau
        tau = self.tau_mlp(F_in).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)

        m = torch.sigmoid((logits - tau) / self.t)

        w_small = a1 + a2  # (B,1,H,W)

        m_so = m * w_small

        return m_so, m, w_small, logits


# -------------------------
# Full RG-RGD depth-refinement network
# -------------------------
class RGRGDDepthRefiner(nn.Module):
    def __init__(
        self,
        base=32,
        patch=8,
        heads=4,
        bp_iters=1,
        sofa_bias_scale=2.0,
        learn_kmap=True,
        proxy_logit_scale: float = 1.5,
        # ViT
        vit_name: str = "vit_small_patch16_224",
        vit_patch: int = 16,
        vit_pretrained: bool = True,
        vit_local_weights: str = "model.safetensors",
        # YOLO
        yolo_mode: str = "none",  # none|online
        yolo_ckpt: str = "yolov8n.pt",
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.7,
        yolo_small_area_max: int = 48 * 48,
        yolo_weight: float = 0.4,
        # Transformer backbone
        win_size: int = 8,
        tf_depth: int = 2,
        # uncertainty init
        init_logvar: float = -2.0,
        # BFS-Head
        bfs_p: float = 0.05,
        bfs_t: float = 0.1,
        # BFS mask stabilization
        mask_blur_ks: int = 7,
        mask_blur_sigma: float = 2.0,
        so_budget_norm: bool = True,

        # BP stability (variance overconfidence fix)
        bp_beta: float = 0.2,
        bp_refine_var: bool = False,
        # output parameterization
        use_residual: bool = True,
        hard_sparse_copy: bool = True,
        max_depth: float = 10.0,
        # uncertainty clamp (for stable NLL)
        min_sigma: float = 0.02,
        max_sigma: float = 10.0,
        # BMF (measurement uncertainty clamp)
        min_sigma_obs: float = 0.01,
        max_sigma_obs: float = 5.0,
        # CSPN-lite refinement (extra geometric propagation)
        cspn_enable: bool = True,
        cspn_iters: int = 3,
        cspn_hidden: int = 16,
        cspn_use_diag: bool = True,
        # Sparse observation handling (IMPORTANT for dense depth cameras)
        obs_keep_prob: float = 0.05,
        obs_max_frac: float = 0.25,
        obs_subsample_when_dense: bool = True,

    ):
        super().__init__()
        self.obs_keep_prob = float(obs_keep_prob)
        self.obs_max_frac = float(obs_max_frac)
        self.obs_subsample_when_dense = bool(obs_subsample_when_dense)
        self.base = int(base)
        self.patch = int(patch)
        self.yolo_mode = str(yolo_mode).lower()
        self.yolo_weight = float(yolo_weight)
        self.mask_blur_ks = int(mask_blur_ks)
        self.mask_blur_sigma = float(mask_blur_sigma)
        self.so_budget_norm = bool(so_budget_norm)


        # output / uncertainty config
        self.use_residual = bool(use_residual)
        self.hard_sparse_copy = bool(hard_sparse_copy)
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_logvar = float(math.log(self.min_sigma ** 2 + 1e-12))
        self.max_logvar = float(math.log(self.max_sigma ** 2 + 1e-12))

        # BMF measurement uncertainty clamp
        self.min_sigma_obs = float(min_sigma_obs)
        self.max_sigma_obs = float(max_sigma_obs)
        self.min_logvar_obs = float(math.log(self.min_sigma_obs ** 2 + 1e-12))
        self.max_logvar_obs = float(math.log(self.max_sigma_obs ** 2 + 1e-12))
        self.bp_refine_var = bool(bp_refine_var)

        # stems
        self.rgb_stem = ViTSRGBStem(
            base=self.base,
            model_name=vit_name,
            patch=vit_patch,
            pretrained=vit_pretrained,
            local_weights=vit_local_weights,
        )

        # ---- Local high-res RGB adapter (cheap) ----
        self.rgb_local = DWConvBNAct(3, self.base, k=3, s=1, p=1)

        # 1-channel gate to mix local detail into ViT feature (very cheap)
        self.rgb_gate = nn.Conv2d(self.base * 2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.rgb_gate.weight)
        nn.init.constant_(self.rgb_gate.bias, -2.0)  # start with small gate (~0.12)
        # depth branch takes: [dense_hint, sparse_depth, validity]
        self.dep_stem = nn.Sequential(ConvBNAct(3, base), ConvBNAct(base, base))

        # High-freq conv for f_hf
        self.hf_conv = ConvBNAct(5, base // 4, 3, 1, 1)  # input: rgb(3) + E(1) + L(1)

        # BFS-Head (replaces so_head)
        bfs_in_ch = base + (base // 4) + 1 + 1 + 1 + 1  # f_rgb + f_hf + depth + vm + E + L
        self.bfs_head = BFSHead(bfs_in_ch, self.base, p=bfs_p, t=bfs_t)

        # SOFA
        self.sofa = SofaCrossAttention(dim=base, heads=heads, patch=patch, bias_scale=sofa_bias_scale)
        self.fuse = ConvBNAct(base * 2, base)

        # transformer backbone
        self.backbone = TransformerUNetBackbone(base=base, heads=heads, ws=win_size, depth=tf_depth)

        # PPE + MIM
        self.ppe = PPE(ch=base, learn_kmap=learn_kmap)
        self.mim = MIM(ch=base)

        # outputs
        self.mu_head = nn.Conv2d(base, 1, 3, 1, 1)
        self.sigma_raw = nn.Conv2d(base, 1, 3, 1, 1)
        # BMF: measurement uncertainty head (sigma_obs)
        # Inputs: y (base), mpi (1), mask_so (1), E (1), vm (1) => base+4
        self.sigma_obs_raw = nn.Conv2d(base + 4, 1, 3, 1, 1)
        self.bp = LiteLearnedPropRefiner(
            iters=bp_iters,
            guide_ch=base,
            hidden=8,
            beta_init=bp_beta,
            refine_var=bp_refine_var,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

        # CSPN-lite (extra edge-aware geometric propagation)
        self.cspn = None
        if bool(cspn_enable) and int(cspn_iters) > 0:
            self.cspn = UACSPNRefiner(
                iters=int(cspn_iters),
                guide_ch=base,
                hidden=int(cspn_hidden),
                use_diag=bool(cspn_use_diag),
            )

        # proxy params (kept for compatibility, but BFS-SOFA replaces much of it)
        self.proxy_edge_thr = nn.Parameter(torch.tensor(0.15))
        self.proxy_dilate = 5
        self.proxy_logit_scale = float(proxy_logit_scale)

        # YOLO (online only)
        if self.yolo_mode == "online":
            self.yolo = YoloMaskGenerator(
                ckpt=yolo_ckpt,
                conf=yolo_conf,
                iou=yolo_iou,
                small_area_max=yolo_small_area_max,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.yolo = None

        # init sigma head for stable NLL
        self._init_uncertainty_head(init_logvar=init_logvar)

    def _init_uncertainty_head(self, init_logvar: float = -2.0):
        # target sigma = exp(0.5*logvar)
        sigma0 = float(math.exp(0.5 * init_logvar))
        # sigma = softplus(bias) + 1e-3 => bias = inv_softplus(sigma0 - 1e-3)
        bias0 = inv_softplus(max(sigma0 - 1e-3, 1e-4))
        nn.init.zeros_(self.sigma_raw.weight)
        nn.init.constant_(self.sigma_raw.bias, bias0)
        # Initialize measurement uncertainty head (sigma_obs) slightly more conservative
        sigma_obs0 = float(math.exp(0.5 * (init_logvar - 0.5)))
        bias_obs0 = inv_softplus(max(sigma_obs0 - 1e-3, 1e-4))
        nn.init.zeros_(self.sigma_obs_raw.weight)
        nn.init.constant_(self.sigma_obs_raw.bias, bias_obs0)

    def compute_high_freq(self, rgb: torch.Tensor):
        gray = rgb.mean(1, keepdim=True)
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 3)
        ex = F.conv2d(gray, kx, padding=1).abs()
        ey = F.conv2d(gray, ky, padding=1).abs()
        E = (ex + ey).clamp(0, 1)
        # Laplacian kernel
        kl = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 3)
        L = F.conv2d(gray, kl, padding=1).abs().clamp(0, 1)
        return E, L

    def build_small_object_proxy(self, rgb: torch.Tensor, vm: torch.Tensor):
        dx = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).mean(1, keepdim=True)
        dy = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).mean(1, keepdim=True)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = (dx + dy).clamp(0, 1)

        thr = torch.sigmoid(self.proxy_edge_thr)
        edge_soft = torch.sigmoid(12.0 * (edge - thr))

        k = self.proxy_dilate
        vm_d = F.max_pool2d(vm, kernel_size=k, stride=1, padding=k // 2)
        proxy = (0.6 * vm_d + 0.4 * edge_soft).clamp(0, 1)
        return proxy

    def forward(self, x, use_so: bool = True, record_fovea_stats: bool = False, attn_thresh: float = 0.5):
        rgb = x[:, :3]
        depth = x[:, 3:4]  # sparse depth
        vm = x[:, 4:5]
        depth_fill = x[:, 5:6] if x.shape[1] > 5 else depth  # optional dense hint

        f_vit = self.rgb_stem(rgb)      # global semantics (coarse)
        f_loc = self.rgb_local(rgb)     # local details (full-res, cheap)
        g_rgb = torch.sigmoid(self.rgb_gate(torch.cat([f_vit, f_loc], dim=1)))
        f_rgb = f_vit + g_rgb * f_loc
        f_dep = self.dep_stem(torch.cat([depth_fill, depth, vm], dim=1))

        # High-freq features
        E, L = self.compute_high_freq(rgb)
        hf_in = torch.cat([rgb, E, L], dim=1)
        f_hf = self.hf_conv(hf_in)

        # BFS input
        F_in = torch.cat([f_rgb, f_hf, depth_fill, vm, E, L], dim=1)

        if use_so:
            # BFS-Head
            mask_so, m, w_small, mask_logits = self.bfs_head(F_in)

            # Optional: blend with proxy (for compatibility, but BFS is primary)
            # Proxy prior (used as a *logit bias*, not as a post-hoc mask mixture)
            proxy = self.build_small_object_proxy(rgb, vm)
            # Convert proxy in [0,1] to a signed logit bias around 0
            proxy_bias = self.proxy_logit_scale * (proxy - 0.5)
            mask_logits = mask_logits + proxy_bias
            # Smooth logits to avoid speckle-like masks (differentiable)
            if self.mask_blur_ks and int(self.mask_blur_ks) > 1:
                mask_logits = gaussian_blur2d(mask_logits, ks=int(self.mask_blur_ks), sigma=float(self.mask_blur_sigma))
            # Recompute mask with biased logits to stay consistent with what SOFA will use
            tau = self.bfs_head.tau_mlp(F_in).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
            m = torch.sigmoid((mask_logits - tau) / float(self.bfs_head.t))
            mask_so = (m * w_small).clamp(0, 1)


            if self.yolo is not None:
                try:
                    mask_yolo = self.yolo(rgb).clamp(0, 1)
                    mask_so = ((1.0 - self.yolo_weight) * mask_so + self.yolo_weight * mask_yolo).clamp(0, 1)
                except Exception:
                    mask_yolo = torch.zeros_like(mask_so)
            else:
                mask_yolo = torch.zeros_like(mask_so)

            # Optional: keep the mask on a stable budget to avoid collapse/speckles
            if self.so_budget_norm and float(self.bfs_head.p) > 0:
                mass = mask_so.mean(dim=(2, 3), keepdim=True)
                scale = (float(self.bfs_head.p) / (mass + 1e-6))
                mask_so = (mask_so * scale).clamp(0, 1)

            # SOFA
            sofa_out = self.sofa(f_rgb, f_dep, mask_so, return_stats=bool(record_fovea_stats), attn_thresh=float(attn_thresh))
            if record_fovea_stats:
                f_add, sofa_stats = sofa_out
            else:
                f_add, sofa_stats = sofa_out, None
            f_rgb = f_rgb + f_add
        else:
            # Baseline: no SOFA, zero masks
            mask_so = torch.zeros_like(vm)
            m = torch.zeros_like(vm)
            w_small = torch.zeros_like(vm)
            mask_logits = torch.zeros_like(vm)
            mask_yolo = torch.zeros_like(vm)
            proxy = torch.zeros_like(vm)

        f_fuse = self.fuse(torch.cat([f_rgb, f_dep], dim=1))

        # transformer backbone
        y = self.backbone(f_fuse)

        # PPE
        y = self.ppe(y, depth_fill)
        # MIM
        y, mpi_mask, sig_lift = self.mim(y, rgb)

        mu_res = self.mu_head(y)
        if self.use_residual:
            mu = depth_fill + mu_res
        else:
            mu = mu_res
        mu = F.relu(mu)
        if self.max_depth is not None and self.max_depth > 0:
            mu = mu.clamp(0.0, float(self.max_depth))

        sigma = F.softplus(self.sigma_raw(y)) + 1e-6
        sigma = torch.clamp(sigma, self.min_sigma, self.max_sigma)
        logvar = 2.0 * torch.log(sigma)
        logvar = (logvar + 0.5 * torch.tanh(sig_lift)).clamp(self.min_logvar, self.max_logvar)
        # ---- BMF: Uncertainty-consistent Bayesian Measurement Fusion ----
        # Estimate measurement uncertainty sigma_obs conditioned on MPI/material cues and foveated mask.
        sigma_obs = F.softplus(self.sigma_obs_raw(torch.cat([y, mpi_mask, mask_so, E, vm], dim=1))) + 1e-6
        sigma_obs = torch.clamp(sigma_obs, self.min_sigma_obs, self.max_sigma_obs)
        logvar_obs = (2.0 * torch.log(sigma_obs)).clamp(self.min_logvar_obs, self.max_logvar_obs)

        if self.hard_sparse_copy:
            # Fuse prediction (mu, logvar) with measurement (depth, logvar_obs) only where measurement exists.
            obs = (vm > 0.5) & (depth > 1e-6)
            var_p = torch.exp(logvar)
            var_o = torch.exp(logvar_obs)
            eps = 1e-12
            prec_p = 1.0 / (var_p + eps)
            prec_o = 1.0 / (var_o + eps)
            prec_sum = prec_p + prec_o
            mu_f = (mu * prec_p + depth * prec_o) / (prec_sum + eps)
            var_f = 1.0 / (prec_sum + eps)
            mu = torch.where(obs, mu_f, mu)
            logvar = torch.where(obs, torch.log(var_f + eps), logvar)
            logvar = logvar.clamp(self.min_logvar, self.max_logvar)

        # anchor observed/fused measurements before refiners.
        # NOTE: This model was originally designed for SPARSE observations.
        # For dense depth cameras, obs can cover most pixels; anchoring with detach everywhere
        # will kill gradients for the entire depth network. We therefore subsample obs during training.
        obs = (vm > 0.5) & (depth > 1e-6)
        if self.training and self.obs_subsample_when_dense and (self.obs_keep_prob < 1.0):
            # if obs is too dense, keep only a random subset
            obs_frac = float(obs.float().mean().item())
            if obs_frac > self.obs_max_frac:
                keep = (torch.rand_like(vm) < self.obs_keep_prob)
                obs = obs & keep
        anchor_mu = mu.detach()

        mu_ref, logvar_ref = self.bp(mu, logvar, rgb, guide=y, mask_so=mask_so)
        # Clamp before CSPN to avoid extreme log-variance values destabilizing the refiner.
        logvar_ref = logvar_ref.clamp(self.min_logvar, self.max_logvar)
        if self.cspn is not None:
            mu_ref = self.cspn(
                mu_ref,
                logvar_ref,
                rgb,
                guide=y,
                mask_so=mask_so,
                vm=vm,
                anchor_mu=anchor_mu,
                obs=obs,
            )
        # logvar_ref already clamped
        aux = {
            "mask_so": mask_so,
            "m": m,  # the m from BFS (formerly mask_pred)
            "w_small": w_small,
            "bfs_logits": mask_logits,  # for BCE in distill/benefit
            "proxy": proxy,
            "mpi": mpi_mask,
            "sigma_obs": sigma_obs,
            "logvar_obs": logvar_obs,
            "yolo": mask_yolo,
            "depth_fill": depth_fill,
        }
        if record_fovea_stats and isinstance(locals().get('sofa_stats', None), dict):
            aux.update(sofa_stats)
        return mu_ref, logvar_ref, aux


# -------------------------
# Loss + distillation + NLL warmup driven by crit.w_nll updated per-epoch
# -------------------------




# Backward-compatible alias for older internal checkpoints/scripts.
SODR_ViT_YOLO_Transformer = RGRGDDepthRefiner

# -----------------------------------------------------------------------------
# RGB-D/IMU self-supervised training utilities
# -----------------------------------------------------------------------------
# The code below implements dataset loading, IMU-assisted pose estimation,
# differentiable view synthesis, benefit-target construction, and the
# optimization loop used by tools/train_rgbd_imu_selfsup.py.
#
# Notes for open-source release:
# - No private paths or data are embedded in this file.
# - Depth frames should be registered to the RGB camera before training.
# - The dense depth hint is used for refinement and warping; users can mask
#   physically invalid background/sky pixels during visualization or evaluation.

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_u16(path: str) -> np.ndarray:
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(path)
    return x


def center_crop_square(img: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """Return (cropped_img, x0, y0, side)."""
    h, w = img.shape[:2]
    side = min(h, w)
    x0 = int((w - side) // 2)
    y0 = int((h - side) // 2)
    return img[y0:y0 + side, x0:x0 + side].copy(), x0, y0, side


def safe_resize(img: np.ndarray, size_hw: Tuple[int, int], interp: int) -> np.ndarray:
    h, w = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(img, (w, h), interpolation=interp)


def fill_sparse_depth_nearest(depth: np.ndarray, vm: np.ndarray) -> np.ndarray:
    """Nearest fill for sparse depth using OpenCV distance transform with labels."""
    depth2 = depth.squeeze().astype(np.float32)
    vm2 = vm.squeeze().astype(np.float32)
    if vm2.max() < 0.5:
        return depth2.copy()
    inv = (vm2 < 0.5).astype(np.uint8)  # invalid=1
    _dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(inv == 0)
    if ys.size == 0:
        return depth2.copy()
    lbls = labels[ys, xs]
    max_lbl = int(labels.max())
    map_y = np.zeros(max_lbl + 1, dtype=np.int32)
    map_x = np.zeros(max_lbl + 1, dtype=np.int32)
    map_y[0] = int(ys[0])
    map_x[0] = int(xs[0])
    map_y[lbls] = ys.astype(np.int32)
    map_x[lbls] = xs.astype(np.int32)
    filled = depth2[map_y[labels], map_x[labels]]
    return filled.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# IMU utilities
# -----------------------------------------------------------------------------
@dataclass
class IMUCache:
    t_us: np.ndarray     # (N,) int64
    acc: np.ndarray      # (N,3) float32 in Color camera frame
    gyro: np.ndarray     # (N,3) float32 in Color camera frame


def parse_imu_R_g2c(values: Optional[List[float]]) -> np.ndarray:
    """Return 3x3 float32 row-major rotation matrix Gyro->Color."""
    if values is None or len(values) == 0:
        # Default from your exported Gyro_to_Color R (row-major):
        return np.array([
            [0.9998027682304382, -0.019803086295723915, -0.0014865585835650563],
            [0.019801009446382523, 0.9998029470443726, -0.0013990221777930856],
            [0.001513970666565001, 0.001369310892187059, 0.9999979138374329],
        ], dtype=np.float32)
    if len(values) != 9:
        raise ValueError("--imu_R_g2c must have 9 floats (row-major).")
    return np.array(values, dtype=np.float32).reshape(3, 3)


def load_imu_csv(csv_path: str, R_g2c: np.ndarray) -> IMUCache:
    """Load IMU csv, merge duplicate timestamps, forward-fill missing, then rotate to camera frame."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    data = None
    try:
        data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float64, skip_header=1)
    except Exception:
        data = None
    if data is None or (isinstance(data, np.ndarray) and (data.size == 0 or np.isnan(data).all() or (data.ndim == 2 and data.shape[1] < 7))):
        data = np.genfromtxt(csv_path, delimiter=None, dtype=np.float64, skip_header=1)

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise RuntimeError(f"imu.csv must have 7 columns: timestamp_us ax ay az gx gy gz. Got shape={data.shape}")

    t_us = data[:, 0].astype(np.int64)
    acc = data[:, 1:4].astype(np.float32)
    gyro = data[:, 4:7].astype(np.float32)

    idx = np.argsort(t_us)
    t_us, acc, gyro = t_us[idx], acc[idx], gyro[idx]

    uniq_t, starts = np.unique(t_us, return_index=True)
    if len(uniq_t) != len(t_us):
        acc_m = np.zeros((len(uniq_t), 3), dtype=np.float32)
        gyro_m = np.zeros((len(uniq_t), 3), dtype=np.float32)
        for i, _t in enumerate(uniq_t):
            j0 = starts[i]
            j1 = starts[i + 1] if i + 1 < len(starts) else len(t_us)
            a = acc[j0:j1]
            g = gyro[j0:j1]
            an = np.linalg.norm(a, axis=1)
            gn = np.linalg.norm(g, axis=1)
            if an.max() > 1e-6:
                acc_m[i] = a[an.argmax()]
            if gn.max() > 1e-6:
                gyro_m[i] = g[gn.argmax()]
        t_us, acc, gyro = uniq_t.astype(np.int64), acc_m, gyro_m

    for i in range(1, len(t_us)):
        if np.linalg.norm(acc[i]) < 1e-8:
            acc[i] = acc[i - 1]
        if np.linalg.norm(gyro[i]) < 1e-8:
            gyro[i] = gyro[i - 1]

    # rotate IMU vectors into Color camera frame
    acc = (R_g2c @ acc.T).T.astype(np.float32, copy=False)
    gyro = (R_g2c @ gyro.T).T.astype(np.float32, copy=False)

    return IMUCache(t_us=t_us, acc=acc, gyro=gyro)


def integrate_gyro_dtheta(imu: IMUCache, t0_us: int, t1_us: int, gyro_unit: str, max_gap_us: int) -> np.ndarray:
    if t1_us <= t0_us:
        return np.zeros((3,), np.float32)
    if (t1_us - t0_us) > int(max_gap_us):
        return np.zeros((3,), np.float32)

    t = imu.t_us
    l = int(np.searchsorted(t, t0_us, side="left"))
    r = int(np.searchsorted(t, t1_us, side="right"))
    if r - l < 2:
        return np.zeros((3,), np.float32)

    tt = imu.t_us[l:r].astype(np.float64) * 1e-6
    g = imu.gyro[l:r].astype(np.float64)
    if gyro_unit.lower().startswith("deg"):
        g = g * (math.pi / 180.0)

    dt = np.diff(tt)
    g_mid = 0.5 * (g[1:] + g[:-1])
    dtheta = (g_mid * dt[:, None]).sum(axis=0)
    return dtheta.astype(np.float32)


def imu_window_stats(imu: IMUCache, t0_us: int, t1_us: int, gyro_unit: str, max_gap_us: int, win_us: int = 20000) -> np.ndarray:
    """12-dim: [dtheta(3), mean_acc(3), mean_gyro(3), std_gyro(3)] in Color camera frame."""
    if t1_us <= t0_us:
        return np.zeros((12,), np.float32)
    if (t1_us - t0_us) > int(max_gap_us):
        return np.zeros((12,), np.float32)

    t = imu.t_us
    l = int(np.searchsorted(t, t1_us - int(win_us), side="left"))
    r = int(np.searchsorted(t, t1_us, side="right"))
    if r - l < 2:
        return np.zeros((12,), np.float32)

    acc = imu.acc[l:r].astype(np.float32)
    gyro = imu.gyro[l:r].astype(np.float32)
    if gyro_unit.lower().startswith("deg"):
        gyro = gyro * (math.pi / 180.0)

    dtheta = integrate_gyro_dtheta(imu, t0_us, t1_us, gyro_unit=gyro_unit, max_gap_us=max_gap_us)
    mean_a = acc.mean(axis=0)
    mean_g = gyro.mean(axis=0)
    std_g = gyro.std(axis=0)

    return np.concatenate([dtheta, mean_a, mean_g, std_g], axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle exp map: (B,3)->(B,3,3)."""
    B = w.shape[0]
    theta = torch.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        zero, -kz,  ky,
         kz, zero, -kx,
        -ky,  kx, zero
    ], dim=1).view(B, 3, 3)

    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(B, 1, 1)
    sin = torch.sin(theta).view(B, 1, 1)
    cos = torch.cos(theta).view(B, 1, 1)
    return I + sin * K + (1 - cos) * (K @ K)


def project(K: Intrinsics, X: torch.Tensor) -> torch.Tensor:
    x = X[:, 0] / X[:, 2].clamp_min(1e-6)
    y = X[:, 1] / X[:, 2].clamp_min(1e-6)
    u = K.fx * x + K.cx
    v = K.fy * y + K.cy
    return torch.stack([u, v], dim=1)


def backproject(K: Intrinsics, depth: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing="ij"
    )
    xs = xs.reshape(1, 1, -1).repeat(B, 1, 1)
    ys = ys.reshape(1, 1, -1).repeat(B, 1, 1)
    z = depth.reshape(B, 1, -1)
    x = (xs - K.cx) / K.fx * z
    y = (ys - K.cy) / K.fy * z
    return torch.cat([x, y, z], dim=1)


def make_grid(u: torch.Tensor, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, N = u.shape
    gx = (u / (W - 1) * 2 - 1).view(B, H, W, 1)
    gy = (v / (H - 1) * 2 - 1).view(B, H, W, 1)
    return torch.cat([gx, gy], dim=3)


def warp_src_to_tgt(src: torch.Tensor, depth_tgt: torch.Tensor, R_tgt_to_src: torch.Tensor, t_tgt_to_src: torch.Tensor, K: Intrinsics):
    B, C, H, W = src.shape
    X = backproject(K, depth_tgt)  # (B,3,N)
    Xs = (R_tgt_to_src @ X) + t_tgt_to_src.view(B, 3, 1)
    uv = project(K, Xs)  # (B,2,N)
    u = uv[:, 0]
    v = uv[:, 1]
    grid = make_grid(u, v, H, W)
    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mask = ((u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))).float().view(B, 1, H, W)
    z = Xs[:, 2].view(B, 1, H, W)
    return warped, mask, z


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def huber(x: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    ax = x.abs()
    quad = torch.clamp(ax, max=delta)
    lin = ax - quad
    return 0.5 * quad * quad / (delta + 1e-12) + lin

@torch.no_grad()
def fit_affine_scale_shift(d_pred: torch.Tensor, d_meas: torch.Tensor, mask: torch.Tensor):
    """Fit per-batch affine a,b: minimize ||a*d_pred + b - d_meas||^2 over masked pixels.
    d_pred,d_meas: (B,1,H,W) in meters. mask: (B,1,H,W) {0,1}.
    Returns: a (B,1,1,1), b (B,1,1,1)
    """
    B = d_pred.shape[0]
    a_list = []
    b_list = []
    for bi in range(B):
        m = mask[bi].reshape(-1)
        if float(m.sum()) < 10:
            a_list.append(torch.ones(1, 1, 1, 1, device=d_pred.device, dtype=d_pred.dtype))
            b_list.append(torch.zeros(1, 1, 1, 1, device=d_pred.device, dtype=d_pred.dtype))
            continue
        x = d_pred[bi].reshape(-1)
        y = d_meas[bi].reshape(-1)
        x = x[m > 0.5]
        y = y[m > 0.5]
        mx = x.mean()
        my = y.mean()
        vx = ((x - mx) ** 2).mean().clamp_min(1e-12)
        cov = ((x - mx) * (y - my)).mean()
        a = (cov / vx).clamp(0.0, 10.0)
        b = (my - a * mx).clamp(-5.0, 5.0)
        a_list.append(a.view(1, 1, 1, 1))
        b_list.append(b.view(1, 1, 1, 1))
    a = torch.cat(a_list, dim=0)
    b = torch.cat(b_list, dim=0)
    return a, b

@torch.no_grad()
def fit_scale_only(d_pred: torch.Tensor, d_meas: torch.Tensor, mask: torch.Tensor):
    """Fit per-batch scale a only: minimize ||a*d_pred - d_meas||^2 (shift=0)."""
    B = d_pred.shape[0]
    a_list = []
    for bi in range(B):
        m = mask[bi].reshape(-1)
        if float(m.sum()) < 10:
            a_list.append(torch.ones(1, 1, 1, 1, device=d_pred.device, dtype=d_pred.dtype))
            continue
        x = d_pred[bi].reshape(-1)
        y = d_meas[bi].reshape(-1)
        x = x[m > 0.5]
        y = y[m > 0.5]
        denom = (x * x).mean().clamp_min(1e-12)
        a = ((x * y).mean() / denom).clamp(0.0, 10.0)
        a_list.append(a.view(1, 1, 1, 1))
    return torch.cat(a_list, dim=0)

def meas_residual(d_pred_aligned: torch.Tensor, d_meas: torch.Tensor, mask: torch.Tensor, args) -> torch.Tensor:
    r = d_pred_aligned - d_meas
    if args.meas_loss_type == "l1":
        per = r.abs()
    elif args.meas_loss_type == "charb":
        per = charbonnier(r)
    else:
        per = huber(r, float(args.huber_delta))
    return (per * mask).sum() / (mask.sum() + 1e-6)


def robust_per_pixel(r: torch.Tensor, loss_type: str = "huber", huber_delta: float = 0.1) -> torch.Tensor:
    """Per-pixel robust penalty ρ(r). Returns a tensor with same shape as r."""
    if loss_type == "l1":
        return r.abs()
    if loss_type == "charb":
        return charbonnier(r)
    # huber
    return huber(r, float(huber_delta))


@torch.no_grad()
def build_benefit_target_from_meas_residual(
    d_base_al: torch.Tensor,
    d_fov_al_det: torch.Tensor,
    d_meas: torch.Tensor,
    meas_mask: torch.Tensor,
    args,
) -> torch.Tensor:
    """Build benefit target in [0,1] from measurement residual improvement.
    All inputs are (B,1,H,W). d_fov_al_det MUST be detached already.
    """
    # per-pixel residuals (robust)
    E_base = robust_per_pixel(d_base_al - d_meas, args.meas_loss_type, float(args.huber_delta))
    E_fov = robust_per_pixel(d_fov_al_det - d_meas, args.meas_loss_type, float(args.huber_delta))
    B = torch.relu(E_base - E_fov) * meas_mask

    # Normalize to [0,1]
    if float(getattr(args, "benefit_beta", 0.0)) > 0:
        tilde_B = torch.clamp(B / float(args.benefit_beta), 0, 1)
    else:
        # Quantile normalization on valid pixels
        vv = (meas_mask > 0.5)
        if vv.any():
            q = torch.quantile(B[vv], float(getattr(args, "benefit_q", 0.9)))
            tilde_B = torch.clamp(B / (q + 1e-6), 0, 1)
        else:
            tilde_B = torch.clamp(B, 0, 1)

    # Optional blur (differentiation not needed; target is detached)
    if int(getattr(args, "benefit_blur_ks", 0)) > 1:
        tilde_B = gaussian_blur2d(
            tilde_B,
            ks=int(args.benefit_blur_ks),
            sigma=float(getattr(args, "benefit_blur_sigma", 1.0)),
        )

    return tilde_B

@torch.no_grad()
def build_benefit_target_from_meas_and_photo(
    d_base_al: torch.Tensor,
    d_fov_al_det: torch.Tensor,
    d_meas: torch.Tensor,
    meas_mask: torch.Tensor,
    photo_base: torch.Tensor,
    photo_fov_det: torch.Tensor,
    photo_mask: torch.Tensor,
    args,
) -> torch.Tensor:
    """Build benefit target in [0,1] from a mixture of:
      - measurement residual improvement where meas_mask==1
      - photometric residual improvement where photo_mask==1 (typically on measurement holes)
    All tensors are (B,1,H,W). *_det inputs MUST be detached already.
    """
    # measurement component (robust depth residual)
    E_base = robust_per_pixel(d_base_al - d_meas, args.meas_loss_type, float(args.huber_delta))
    E_fov = robust_per_pixel(d_fov_al_det - d_meas, args.meas_loss_type, float(args.huber_delta))
    B_meas = torch.relu(E_base - E_fov) * meas_mask

    # photo component (raw photometric error, smaller is better)
    # Expect photo_* already in same scale as training photo loss per-pixel term.
    B_photo = torch.relu(photo_base - photo_fov_det) * photo_mask

    # mix
    w_photo = float(getattr(args, "benefit_photo_w", 0.5))
    B = B_meas + w_photo * B_photo

    # union mask for normalization
    um = ((meas_mask > 0.5) | (photo_mask > 0.5)).float()

    # Normalize to [0,1]
    if float(getattr(args, "benefit_beta", 0.0)) > 0:
        tilde_B = torch.clamp(B / float(args.benefit_beta), 0, 1)
    else:
        vv = (um > 0.5)
        if vv.any():
            q = torch.quantile(B[vv], float(getattr(args, "benefit_q", 0.9)))
            tilde_B = torch.clamp(B / (q + 1e-6), 0, 1)
        else:
            tilde_B = torch.clamp(B, 0, 1)

    # Optional blur
    if int(getattr(args, "benefit_blur_ks", 0)) > 1:
        tilde_B = gaussian_blur2d(
            tilde_B,
            ks=int(args.benefit_blur_ks),
            sigma=float(getattr(args, "benefit_blur_sigma", 1.0)),
        )

    # Ensure we never supervise outside union mask
    tilde_B = tilde_B * um
    return tilde_B

def ssim(x: torch.Tensor, y: torch.Tensor):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sig_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sig_x + sig_y + C2)
    out = ssim_n / ssim_d.clamp_min(1e-6)
    return torch.clamp((1 - out) / 2, 0, 1).mean(dim=1, keepdim=True)


def edge_aware_smoothness(depth: torch.Tensor, img: torch.Tensor, mask: torch.Tensor):
    """Edge-aware smoothness, masked to valid depth pixels."""
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-10.0 * img_dx)
    wy = torch.exp(-10.0 * img_dy)
    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    sx = (wx * torch.abs(dx) * mx).sum() / (mx.sum() + 1e-6)
    sy = (wy * torch.abs(dy) * my).sum() / (my.sum() + 1e-6)
    return sx + sy


# -----------------------------------------------------------------------------
# Pose net
# -----------------------------------------------------------------------------
class PoseNet(nn.Module):
    def __init__(self, imu_dim: int = 12, max_trans: float = 0.20, max_rot: float = 0.20):
        super().__init__()
        self.imu_dim = imu_dim
        self.max_trans = float(max_trans)
        self.max_rot = float(max_rot)
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 7, 2, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256 + imu_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, rgb0: torch.Tensor, rgb1: torch.Tensor, imu_feat: torch.Tensor):
        x = torch.cat([rgb0, rgb1], dim=1)
        f = self.enc(x)
        f = self.pool(f).flatten(1)
        if imu_feat is None:
            imu_feat = torch.zeros((f.shape[0], self.imu_dim), device=f.device, dtype=f.dtype)
        if imu_feat.shape[1] != self.imu_dim:
            if imu_feat.shape[1] > self.imu_dim:
                imu_feat = imu_feat[:, : self.imu_dim]
            else:
                pad = torch.zeros((imu_feat.shape[0], self.imu_dim - imu_feat.shape[1]), device=f.device, dtype=f.dtype)
                imu_feat = torch.cat([imu_feat, pad], dim=1)
        y = self.fc(torch.cat([f, imu_feat], dim=1))
        t = torch.tanh(y[:, 0:3]) * self.max_trans
        r = torch.tanh(y[:, 3:6]) * self.max_rot
        return t, r


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# YOLO txt -> ROI weight mask (online)
# -----------------------------------------------------------------------------
def _parse_yolo_classes(s: str):
    s = (s or "").strip()
    if not s:
        return None
    out = set()
    for tok in re.split(r"[,\s]+", s):
        tok = tok.strip()
        if tok == "":
            continue
        out.add(int(tok))
    return out

def _read_yolo_txt(txt_path: str):
    # Return list of (cls, xc, yc, w, h) with normalized coords.
    if (not txt_path) or (not os.path.exists(txt_path)):
        return []
    boxes = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc = float(parts[1]); yc = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
                boxes.append((cls, xc, yc, bw, bh))
    except Exception:
        return []
    return boxes

def _yolo_box_to_pixels(xc, yc, bw, bh, W, H, expand=0.0):
    # normalized -> pixel (x1,y1,x2,y2) in original image coords
    bw = max(0.0, float(bw)) * (1.0 + float(expand))
    bh = max(0.0, float(bh)) * (1.0 + float(expand))
    xc = float(xc); yc = float(yc)
    x1 = (xc - bw / 2.0) * W
    y1 = (yc - bh / 2.0) * H
    x2 = (xc + bw / 2.0) * W
    y2 = (yc + bh / 2.0) * H
    return x1, y1, x2, y2

def _transform_box_to_train_res(x1, y1, x2, y2, orig_w, orig_h, out_w, out_h, preprocess: str):
    # Returns (x1,y1,x2,y2) in output pixel coords
    if preprocess == "resize":
        sx = float(out_w) / float(orig_w)
        sy = float(out_h) / float(orig_h)
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    if preprocess == "crop_square":
        side = min(int(orig_w), int(orig_h))
        ox = (float(orig_w) - float(side)) / 2.0
        oy = (float(orig_h) - float(side)) / 2.0
        # crop
        x1c = x1 - ox; x2c = x2 - ox
        y1c = y1 - oy; y2c = y2 - oy
        sx = float(out_w) / float(side)
        sy = float(out_h) / float(side)
        return x1c * sx, y1c * sy, x2c * sx, y2c * sy

    raise ValueError(f"Unknown preprocess: {preprocess}")

def make_yolo_roi_weight(
    label_dir: str,
    img_path: str,
    orig_w: int, orig_h: int,
    out_w: int, out_h: int,
    preprocess: str,
    cls_filter,
    expand: float,
    bg_alpha: float,
):
    # Return (out_h,out_w) float32 weights in [bg_alpha,1]. If no labels, returns ones.
    if not label_dir:
        return np.ones((out_h, out_w), np.float32)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, stem + ".txt")
    boxes = _read_yolo_txt(txt_path)
    if cls_filter is not None:
        boxes = [b for b in boxes if b[0] in cls_filter]
    if len(boxes) == 0:
        return np.ones((out_h, out_w), np.float32)

    bg_alpha = float(bg_alpha)
    bg_alpha = max(0.0, min(1.0, bg_alpha))
    w = np.full((out_h, out_w), bg_alpha, dtype=np.float32)

    for (_cls, xc, yc, bw, bh) in boxes:
        x1, y1, x2, y2 = _yolo_box_to_pixels(xc, yc, bw, bh, orig_w, orig_h, expand=expand)
        x1o, y1o, x2o, y2o = _transform_box_to_train_res(x1, y1, x2, y2, orig_w, orig_h, out_w, out_h, preprocess)
        xi1 = int(np.floor(min(x1o, x2o))); xi2 = int(np.ceil(max(x1o, x2o)))
        yi1 = int(np.floor(min(y1o, y2o))); yi2 = int(np.ceil(max(y1o, y2o)))
        xi1 = max(0, min(out_w - 1, xi1))
        xi2 = max(0, min(out_w, xi2))
        yi1 = max(0, min(out_h - 1, yi1))
        yi2 = max(0, min(out_h, yi2))
        if xi2 > xi1 and yi2 > yi1:
            w[yi1:yi2, xi1:xi2] = 1.0
    return w

class RecorderPairDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        img_hw: Tuple[int, int],
        rgb_dir: str = "",
        depth_dir: str = "",
        imu_csv: str = "",
        depth_scale: float = 1000.0,
        frame_stride: int = 1,
        pair_mode: str = "jump",
        max_gap_us: int = 200000,
        gyro_unit: str = "rad",
        use_sd_fill: bool = True,
        preprocess: str = "crop_square",
        limit_pairs: int = 0,
        imu_R_g2c: Optional[List[float]] = None,
        yolo_label_dir: str = "",
        yolo_classes: str = "",
        yolo_expand: float = 0.0,
        yolo_bg_alpha: float = 0.10,
    ):
        super().__init__()
        self.data_root = str(data_root or "")
        self.rgb_dir = str(rgb_dir or "")
        self.depth_dir = str(depth_dir or "")
        self.imu_csv = str(imu_csv or "")
        self.H, self.W = int(img_hw[0]), int(img_hw[1])
        self.depth_scale = float(depth_scale)
        self.frame_stride = max(1, int(frame_stride))
        self.pair_mode = str(pair_mode or "jump").lower().strip()
        if self.pair_mode not in ("jump","sliding"):
            self.pair_mode = "jump"
        self.max_gap_us = int(max_gap_us)
        self.gyro_unit = str(gyro_unit)
        self.use_sd_fill = bool(use_sd_fill)
        self.preprocess = str(preprocess)
        self.yolo_label_dir = str(yolo_label_dir or "")
        self.yolo_classes = _parse_yolo_classes(yolo_classes)
        self.yolo_expand = float(yolo_expand)
        self.yolo_bg_alpha = float(yolo_bg_alpha)

        self.R_g2c = parse_imu_R_g2c(imu_R_g2c)

        self.seq_dirs: List[str] = []
        self._direct_mode = False
        if self.rgb_dir and self.depth_dir and self.imu_csv:
            self._direct_mode = True
            self.seq_dirs = [os.path.dirname(os.path.abspath(self.rgb_dir))]
        else:
            if not self.data_root:
                raise RuntimeError("Either set --data_root (containing seq_*), or provide --rgb_dir --depth_dir --imu_csv.")
            self.seq_dirs = sorted([p for p in glob.glob(os.path.join(self.data_root, "seq_*")) if os.path.isdir(p)])
        if len(self.seq_dirs) == 0:
            raise RuntimeError("No usable sequence found. Check paths.")

        self.imu_cache: Dict[str, IMUCache] = {}

        pairs: List[Tuple[str, int, int, str, str, str, str]] = []
        for sd in self.seq_dirs:
            cam_dir = self.rgb_dir if self._direct_mode else os.path.join(sd, "cam0")
            dep_dir = self.depth_dir if self._direct_mode else os.path.join(sd, "depth0")
            if not (os.path.isdir(cam_dir) and os.path.isdir(dep_dir)):
                continue

            rgb_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(cam_dir, "*.png")))
            if len(rgb_files) < 2:
                continue

            frames: List[Tuple[int, str, str]] = []
            for rp in rgb_files:
                stem = os.path.splitext(os.path.basename(rp))[0]
                try:
                    ts = int(stem)
                except Exception:
                    continue
                dp = os.path.join(dep_dir, stem + ".png")
                if os.path.isfile(dp):
                    frames.append((ts, rp, dp))
            frames.sort(key=lambda x: x[0])
            if len(frames) < 2:
                continue

            step_i = self.frame_stride if self.pair_mode == "jump" else 1
            for i in range(0, len(frames) - self.frame_stride, step_i):
                ts0, rp0, dp0 = frames[i]
                ts1, rp1, dp1 = frames[i + self.frame_stride]
                if (ts1 - ts0) <= 0 or (ts1 - ts0) > self.max_gap_us:
                    continue
                pairs.append((sd, ts0, ts1, rp0, dp0, rp1, dp1))

        if len(pairs) == 0:
            raise RuntimeError("No valid consecutive pairs found.")

        if limit_pairs and limit_pairs > 0 and len(pairs) > int(limit_pairs):
            pairs = pairs[: int(limit_pairs)]

        self.pairs = pairs
        print(f"[dataset] sequences={len(self.seq_dirs)} pairs={len(self.pairs)} pair_mode={self.pair_mode} stride={self.frame_stride} preprocess={self.preprocess} -> {self.W}x{self.H}")
        print(f"[imu] Gyro->Color R:\n{self.R_g2c}")

    def __len__(self):
        return len(self.pairs)

    def _get_imu(self, seq_dir: str) -> IMUCache:
        if seq_dir not in self.imu_cache:
            path = self.imu_csv if (self._direct_mode and self.imu_csv) else os.path.join(seq_dir, "imu.csv")
            self.imu_cache[seq_dir] = load_imu_csv(path, self.R_g2c)
        return self.imu_cache[seq_dir]

    def _preprocess_rgb_depth(self, rgb: np.ndarray, dep_m: np.ndarray):
        # If depth resolution differs, resize depth to RGB resolution first (assumes already registered)
        rh, rw = rgb.shape[:2]
        dh, dw = dep_m.shape[:2]
        if (dh != rh) or (dw != rw):
            dep_m = cv2.resize(dep_m, (rw, rh), interpolation=cv2.INTER_NEAREST)

        vm = ((dep_m > 0) & (dep_m < (65000.0 / self.depth_scale))).astype(np.float32)

        if self.preprocess == "crop_square":
            rgb, x0, y0, side = center_crop_square(rgb)
            dep_m = dep_m[y0:y0 + side, x0:x0 + side].copy()
            vm = vm[y0:y0 + side, x0:x0 + side].copy()
            rgb = safe_resize(rgb, (self.H, self.W), cv2.INTER_LINEAR)
            dep_m = safe_resize(dep_m, (self.H, self.W), cv2.INTER_NEAREST)
            vm = safe_resize(vm, (self.H, self.W), cv2.INTER_NEAREST)
        elif self.preprocess == "resize":
            rgb = safe_resize(rgb, (self.H, self.W), cv2.INTER_LINEAR)
            dep_m = safe_resize(dep_m, (self.H, self.W), cv2.INTER_NEAREST)
            vm = safe_resize(vm, (self.H, self.W), cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unknown preprocess: {self.preprocess}")

        vm = (vm > 0.5).astype(np.float32)
        return rgb, dep_m.astype(np.float32), vm.astype(np.float32)

    def _make_x(self, rgb: np.ndarray, dep_m: np.ndarray, vm: np.ndarray) -> torch.Tensor:
        if self.use_sd_fill:
            dep_fill = fill_sparse_depth_nearest(dep_m, vm)
            dep_fill_t = torch.from_numpy(dep_fill).float().unsqueeze(0)
        else:
            dep_fill_t = None

        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        dep_t = torch.from_numpy(dep_m).float().unsqueeze(0)
        vm_t = torch.from_numpy(vm).float().unsqueeze(0)
        if dep_fill_t is not None:
            x = torch.cat([rgb_t, dep_t, vm_t, dep_fill_t], dim=0)  # 6,H,W
        else:
            x = torch.cat([rgb_t, dep_t, vm_t], dim=0)              # 5,H,W
        return x

    def __getitem__(self, idx: int):
        seq_dir, ts0, ts1, rp0, dp0, rp1, dp1 = self.pairs[idx]
        imu = self._get_imu(seq_dir)

        rgb0 = read_rgb(rp0)
        rgb1 = read_rgb(rp1)
        d0 = read_u16(dp0).astype(np.float32) / self.depth_scale
        d1 = read_u16(dp1).astype(np.float32) / self.depth_scale
        # YOLO ROI weight masks (in output/preprocessed resolution)
        orig_h0, orig_w0 = rgb0.shape[:2]
        orig_h1, orig_w1 = rgb1.shape[:2]
        # If --yolo_label_dir is not provided, default to <seq_dir>/labels for multi-scene training.
        label_dir = self.yolo_label_dir
        if (not label_dir) and seq_dir:
            cand = os.path.join(seq_dir, "labels")
            if os.path.isdir(cand):
                label_dir = cand

        roi0 = make_yolo_roi_weight(
            label_dir, rp0, orig_w0, orig_h0, self.W, self.H, self.preprocess,
            self.yolo_classes, self.yolo_expand, self.yolo_bg_alpha,
        )
        roi1 = make_yolo_roi_weight(
            label_dir, rp1, orig_w1, orig_h1, self.W, self.H, self.preprocess,
            self.yolo_classes, self.yolo_expand, self.yolo_bg_alpha,
        )


        rgb0, d0, vm0 = self._preprocess_rgb_depth(rgb0, d0)
        rgb1, d1, vm1 = self._preprocess_rgb_depth(rgb1, d1)

        x0 = self._make_x(rgb0, d0, vm0)
        x1 = self._make_x(rgb1, d1, vm1)

        imu_feat = imu_window_stats(imu, ts0, ts1, gyro_unit=self.gyro_unit, max_gap_us=self.max_gap_us)
        dtheta = imu_feat[:3].copy()

        imu_feat_t = torch.from_numpy(imu_feat).float()
        dtheta_t = torch.from_numpy(dtheta).float()

        d0_t = torch.from_numpy(d0).float().unsqueeze(0)
        d1_t = torch.from_numpy(d1).float().unsqueeze(0)
        vm0_t = torch.from_numpy(vm0).float().unsqueeze(0)
        vm1_t = torch.from_numpy(vm1).float().unsqueeze(0)
        roi0_t = torch.from_numpy(roi0).float().unsqueeze(0)
        roi1_t = torch.from_numpy(roi1).float().unsqueeze(0)


        return x0, x1, d0_t, d1_t, vm0_t, vm1_t, imu_feat_t, dtheta_t, roi0_t, roi1_t


# -----------------------------------------------------------------------------
# Intrinsics conversion
# -----------------------------------------------------------------------------
def compute_K_at_train_res(
    orig_fx: float, orig_fy: float, orig_cx: float, orig_cy: float,
    orig_w: int, orig_h: int,
    out_w: int, out_h: int,
    preprocess: str,
) -> Intrinsics:
    if preprocess == "resize":
        sx = float(out_w) / float(orig_w)
        sy = float(out_h) / float(orig_h)
        return Intrinsics(
            fx=orig_fx * sx,
            fy=orig_fy * sy,
            cx=orig_cx * sx,
            cy=orig_cy * sy,
        )
    if preprocess == "crop_square":
        side = min(int(orig_w), int(orig_h))
        x0 = (int(orig_w) - side) / 2.0
        y0 = (int(orig_h) - side) / 2.0
        cx2 = orig_cx - x0
        cy2 = orig_cy - y0
        s_x = float(out_w) / float(side)
        s_y = float(out_h) / float(side)
        return Intrinsics(
            fx=orig_fx * s_x,
            fy=orig_fy * s_y,
            cx=cx2 * s_x,
            cy=cy2 * s_y,
        )
    raise ValueError(f"Unknown preprocess: {preprocess}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Training (with train/val split + best checkpoint)
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.set_defaults(obs_subsample_when_dense=True)

    # data
    ap.add_argument("--data_root", type=str, default="", help="Root containing seq_*; optional if --rgb_dir/--depth_dir/--imu_csv provided.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rgb_dir", type=str, default="", help="Path to cam0 folder")
    ap.add_argument("--depth_dir", type=str, default="", help="Path to depth0 folder")
    ap.add_argument("--imu_csv", type=str, default="", help="Path to imu.csv")
    ap.add_argument("--img_h", type=int, default=320)
    ap.add_argument("--img_w", type=int, default=320)
    ap.add_argument("--depth_scale", type=float, default=1000.0)

    # preprocessing + intrinsics
    ap.add_argument("--preprocess", type=str, default="crop_square", choices=["crop_square", "resize"])
    ap.add_argument("--intrinsics_mode", type=str, default="auto_rgb", choices=["auto_rgb", "manual"])
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--rgb_fx", type=float, default=1085.184)
    ap.add_argument("--rgb_fy", type=float, default=1085.184)
    ap.add_argument("--rgb_cx", type=float, default=643.5)
    ap.add_argument("--rgb_cy", type=float, default=361.5)
    ap.add_argument("--rgb_width", type=int, default=1280)
    ap.add_argument("--rgb_height", type=int, default=720)

    # dataset pairing + imu
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--pair_mode", type=str, default="jump", choices=["jump","sliding"], help="Pair sampling: jump uses (0,s),(s,2s)...; sliding uses (0,s),(1,1+s)...")
    ap.add_argument("--max_pair_gap_us", type=int, default=200000)
    ap.add_argument("--imu_gyro_unit", type=str, default="rad", choices=["rad", "deg"])
    ap.add_argument("--use_sd_fill", action="store_true")
    ap.add_argument("--limit_pairs", type=int, default=0)
    ap.add_argument("--imu_R_g2c", type=float, nargs=9, default=None)

    # YOLO ROI (online txt)
    ap.add_argument("--yolo_label_dir", type=str, default="", help="Folder containing YOLO txt labels (same stem as RGB).")
    ap.add_argument("--yolo_classes", type=str, default="", help="Optional: class ids to keep, e.g. '0' or '0,1'. Empty=all.")
    ap.add_argument("--yolo_expand", type=float, default=0.0, help="Expand bbox w/h by this ratio (e.g. 0.1 => +10%).")
    ap.add_argument("--yolo_bg_alpha", type=float, default=0.10, help="Weight outside boxes (0..1). Inside boxes weight=1.")

    # train/val split
    ap.add_argument("--val_ratio", type=float, default=0.0, help="0 disables validation split; otherwise split pairs into train/val.")
    ap.add_argument("--split_mode", type=str, default="frame", choices=["frame", "scene"], help="scene splits by seq_dir when possible.")

    # model
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--win_size", type=int, default=8)
    ap.add_argument("--tf_depth", type=int, default=2)
    ap.add_argument("--max_depth", type=float, default=10.0)
    ap.add_argument("--vit_name", type=str, default="dino_vits8")
    ap.add_argument("--vit_patch", type=int, default=16)
    ap.add_argument("--vit_pretrained", action="store_true")
    ap.add_argument("--vit_local_weights", type=str, default="")

    # BFS-SOFA benefit supervision + regularization
    ap.add_argument("--so_warmup_epochs", type=int, default=5, help="Train without SOFA/BFS for first N epochs.")
    ap.add_argument("--benefit_every", type=int, default=4, help="Compute benefit supervision every N iterations (1=every iter).")
    ap.add_argument("--benefit_beta", type=float, default=0.15, help="Normalization scale for benefit target; <=0 uses quantile.")
    ap.add_argument("--benefit_q", type=float, default=0.90, help="Quantile for benefit normalization when --benefit_beta<=0.")
    ap.add_argument("--benefit_blur_ks", type=int, default=7, help="Blur kernel size for benefit target (0/1 disables).")
    ap.add_argument("--benefit_blur_sigma", type=float, default=2.0, help="Blur sigma for benefit target.")
    ap.add_argument("--benefit_use_photo", action="store_true",
               help="Include photometric residual improvement into benefit target (enables supervision on depth holes where vm_meas==0 but warp is valid).")
    ap.add_argument("--benefit_photo_w", type=float, default=0.5,
               help="Weight of photo benefit relative to measurement benefit.")
    ap.add_argument("--benefit_photo_on_all", action="store_true",
               help="Apply photo benefit on all warp-valid pixels (default: only where measurement is invalid).")
    ap.add_argument("--w_benefit", type=float, default=0.10, help="Weight for BFS benefit BCE loss.")

    # BFS mask regularizers (do not require GT)
    ap.add_argument("--bfs_p", type=float, default=0.05, help="Target mask mass ratio (used by mass regularizer).")
    ap.add_argument("--bfs_t", type=float, default=0.10, help="BFS temperature (mask sharpness).")
    ap.add_argument("--w_mass", type=float, default=0.30, help="Mask mass regularizer weight.")
    ap.add_argument("--w_edgetv", type=float, default=0.015, help="Edge-aware TV regularizer weight on BFS mask.")
    ap.add_argument("--w_scale_bfs", type=float, default=0.04, help="Regularize w_small (encourage using small-object scale).")
    ap.add_argument("--w_mask_coh", type=float, default=0.0, help="Local coherence regularizer weight for BFS mask.")
    ap.add_argument("--coh_ks", type=int, default=7, help="Kernel size for local coherence loss.")

    # BFS mask smoothing inside model (logit blur)
    ap.add_argument("--mask_blur_ks", type=int, default=7)
    ap.add_argument("--mask_blur_sigma", type=float, default=2.0)
    ap.add_argument("--obs_keep_prob", type=float, default=1.0)
    ap.add_argument("--obs_max_frac", type=float, default=1.0)
    ap.add_argument("--obs_subsample_when_dense", action="store_true")

    # residual refinement option (keep compatible with your checkpoints)
    ap.add_argument("--residual_mode", action="store_true")
    ap.add_argument("--res_alpha", type=float, default=0.1)
    ap.add_argument("--delta_max", type=float, default=0.5)

    # pose
    ap.add_argument("--pose_imu_rot", action="store_true")
    ap.add_argument("--pose_max_trans", type=float, default=0.25)
    ap.add_argument("--pose_max_rot", type=float, default=0.35)

    # losses
    ap.add_argument("--w_photo", type=float, default=1.0)
    ap.add_argument("--w_geo", type=float, default=0.5)
    ap.add_argument("--w_meas", type=float, default=0.03)
    ap.add_argument("--meas_sparse", action="store_true",
                   help="Subsample measurement supervision mask when dense (to avoid over-constraining refinement).")
    ap.add_argument("--meas_keep_prob", type=float, default=None,
                   help="When --meas_sparse: keep probability for measured pixels (default: use --obs_keep_prob).")
    ap.add_argument("--meas_max_frac", type=float, default=None,
                   help="When --meas_sparse: only subsample if mask density exceeds this fraction (default: use --obs_max_frac).")
    ap.add_argument("--w_smooth", type=float, default=0.001)

    # --- v10: geometry/warp decoupling + affine-aligned depth supervision ---
    ap.add_argument("--w_warp", type=float, default=0.0,
               help="Weight for geometry-only warp consistency loss (uses depth+pose, no RGB mask).")
    ap.add_argument("--geo_photo_beta", type=float, default=0.0,
               help="Geo loss will be weighted by exp(-beta * photo_error) to make it more RGB-reliable (detach).")
    ap.add_argument("--meas_align", type=str, default="none", choices=["none", "scale", "affine"],
               help="Align predicted depth to measured depth per-batch before computing L_meas.")
    ap.add_argument("--w_shift_reg", type=float, default=1e-3,
               help="Regularize affine shift b (meters) to suppress global bias (applied when meas_align=affine).")
    ap.add_argument("--w_scale_reg", type=float, default=1e-4,
               help="Regularize affine scale a towards 1.0 (applied when meas_align in {scale,affine}).")
    ap.add_argument("--meas_loss_type", type=str, default="huber", choices=["l1", "huber", "charb"],
               help="Depth measurement loss type after alignment.")
    ap.add_argument("--huber_delta", type=float, default=0.10,
               help="Huber delta (meters) for meas_loss_type=huber.")
    ap.add_argument("--w_ssim", type=float, default=0.85)
    ap.add_argument("--occ_thresh", type=float, default=0.05)
    ap.add_argument("--auto_mask", action="store_true")
    ap.add_argument("--warp_eps", type=float, default=1e-3)

    # optimization
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_depth", type=float, default=None)
    ap.add_argument("--lr_pose", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    _nw_default = 0 if os.name == "nt" else 4
    ap.add_argument("--num_workers", type=int, default=_nw_default)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (saves VRAM). Default: on for CUDA unless --no_amp.")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision even on CUDA.")
    ap.add_argument("--vis_every", type=int, default=0, help="Save training visualizations every N batches (0 disables).")
    ap.add_argument("--vis_max_per_epoch", type=int, default=8, help="Max visualization samples to save per epoch per split.")
    ap.add_argument("--vis_split", type=str, default="train", choices=["train","val","both"], help="Which split to save visualizations for.")
    ap.add_argument("--vis_save_npz", action="store_true", help="Also save raw arrays (.npz) for paper/analysis.")
    ap.add_argument("--vis_overlay_alpha", type=float, default=0.45, help="Alpha for overlaying masks on RGB.")
    ap.add_argument("--photo_vis_max", type=float, default=0.30, help="Max value for photo error visualization scaling.")
    ap.add_argument("--geo_vis_max", type=float, default=1.0, help="Max value for geo error visualization scaling.")

    # saving
    ap.add_argument("--save_all", action="store_true", help="If set, also save ckpt_epXXX.pth each epoch.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--print_trainables", action="store_true")

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Safety: residual_mode and per-batch affine alignment can conflict.
    # If you allow an affine (a*d+b) fit before meas_loss, the fit can
    # absorb global shift/scale, making meas_loss blind to the very bias
    # residual_mode tends to exploit (pushing residual to +/- delta_max).
    # ------------------------------------------------------------------
    if getattr(args, "residual_mode", False) and getattr(args, "meas_align", "none") == "affine":
        print("[warn] residual_mode + meas_align=affine can nullify global shift/scale supervision; "
              "overriding meas_align -> none. (Use --meas_align scale/none.)")
        args.meas_align = "none"
    if getattr(args, "meas_align", "none") == "none" and getattr(args, "w_meas", 0.0) > 0.2:
        print(f"[warn] meas_align=none with w_meas={args.w_meas} may dominate optimization; "
              "consider --w_meas 0.05~0.1.")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Mixed precision (saves VRAM). Default: enabled on CUDA unless --no_amp.
    if bool(getattr(args, "no_amp", False)):
        use_amp = False
    elif bool(getattr(args, "amp", False)):
        use_amp = True
    else:
        use_amp = (device.type == "cuda")
    if use_amp and device.type != "cuda":
        use_amp = False
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    H, W = int(args.img_h), int(args.img_w)

    # intrinsics
    if args.intrinsics_mode == "manual":
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            raise RuntimeError("intrinsics_mode=manual requires --fx --fy --cx --cy")
        K = Intrinsics(fx=float(args.fx), fy=float(args.fy), cx=float(args.cx), cy=float(args.cy))
        print(f"[K manual] fx={K.fx:.3f} fy={K.fy:.3f} cx={K.cx:.3f} cy={K.cy:.3f} (at {W}x{H})")
    else:
        K = compute_K_at_train_res(
            orig_fx=float(args.rgb_fx), orig_fy=float(args.rgb_fy),
            orig_cx=float(args.rgb_cx), orig_cy=float(args.rgb_cy),
            orig_w=int(args.rgb_width), orig_h=int(args.rgb_height),
            out_w=W, out_h=H,
            preprocess=str(args.preprocess),
        )
        print(f"[K auto_rgb] preprocess={args.preprocess} orig={args.rgb_width}x{args.rgb_height} -> {W}x{H}")
        print(f"[K auto_rgb] fx={K.fx:.3f} fy={K.fy:.3f} cx={K.cx:.3f} cy={K.cy:.3f}")

    # dataset
    ds_full = RecorderPairDataset(
        data_root=args.data_root,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        imu_csv=args.imu_csv,
        img_hw=(H, W),
        depth_scale=args.depth_scale,
        frame_stride=args.frame_stride,
        pair_mode=args.pair_mode,
        max_gap_us=args.max_pair_gap_us,
        gyro_unit=args.imu_gyro_unit,
        use_sd_fill=bool(args.use_sd_fill),
        preprocess=str(args.preprocess),
        limit_pairs=args.limit_pairs,
        imu_R_g2c=args.imu_R_g2c,
        yolo_label_dir=args.yolo_label_dir,
        yolo_classes=args.yolo_classes,
        yolo_expand=args.yolo_expand,
        yolo_bg_alpha=args.yolo_bg_alpha,
    )

    # split train/val
    def _split_indices(pairs, val_ratio: float, mode: str, seed: int):
        n = len(pairs)
        idx = list(range(n))
        if val_ratio <= 0 or n < 2:
            return idx, []
        rng = np.random.RandomState(int(seed))
        mode = str(mode)
        if mode == "scene":
            seqs = sorted(set([p[0] for p in pairs]))
            if len(seqs) > 1:
                rng.shuffle(seqs)
                n_val = max(1, int(round(len(seqs) * float(val_ratio))))
                val_seqs = set(seqs[:n_val])
                va = [i for i,p in enumerate(pairs) if p[0] in val_seqs]
                tr = [i for i in idx if i not in set(va)]
                if len(tr) == 0 or len(va) == 0:
                    # fallback
                    mode = "frame"
                else:
                    return tr, va
            else:
                mode = "frame"
        # frame split fallback
        rng.shuffle(idx)
        n_val = max(1, int(round(n * float(val_ratio))))
        va = sorted(idx[:n_val])
        tr = sorted(idx[n_val:])
        if len(tr) == 0:
            tr, va = idx, []
        return tr, va

    tr_idx, va_idx = _split_indices(ds_full.pairs, float(args.val_ratio), args.split_mode, args.seed)
    if len(va_idx) > 0:
        ds_tr = Subset(ds_full, tr_idx)
        ds_va = Subset(ds_full, va_idx)
        print(f"[split] mode={args.split_mode} train_pairs={len(tr_idx)} val_pairs={len(va_idx)}")
    else:
        ds_tr = ds_full
        ds_va = None
        print(f"[split] disabled (val_ratio={args.val_ratio}) train_pairs={len(ds_full)}")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    dl_va = None
    if ds_va is not None:
        dl_va = DataLoader(
            ds_va,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=True,
            drop_last=False,
        )

    # models
    depth_net = RGRGDDepthRefiner(
        base=int(args.base),
        patch=int(args.patch),
        heads=int(args.heads),
        win_size=int(args.win_size),
        tf_depth=int(args.tf_depth),
        max_depth=float(args.max_depth),
        yolo_mode="none",
        vit_name=str(args.vit_name),
        vit_patch=int(args.vit_patch),
        vit_pretrained=bool(args.vit_pretrained),
        obs_keep_prob=float(args.obs_keep_prob),
        obs_max_frac=float(args.obs_max_frac),
        obs_subsample_when_dense=bool(args.obs_subsample_when_dense),
        vit_local_weights=str(args.vit_local_weights),
        bfs_p=float(args.bfs_p),
        bfs_t=float(args.bfs_t),
        mask_blur_ks=int(args.mask_blur_ks),
        mask_blur_sigma=float(args.mask_blur_sigma),
    ).to(device)

    pose_net = PoseNet(imu_dim=12, max_trans=float(args.pose_max_trans), max_rot=float(args.pose_max_rot)).to(device)

    if bool(args.print_trainables):
        def _count_trainable(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def _count_all(m):
            return sum(p.numel() for p in m.parameters())
        print(f"[trainables] depth_net trainable={_count_trainable(depth_net):,} / total={_count_all(depth_net):,}")
        print(f"[trainables] pose_net  trainable={_count_trainable(pose_net):,} / total={_count_all(pose_net):,}")

    lr_depth = float(args.lr_depth) if args.lr_depth is not None else float(args.lr)
    lr_pose  = float(args.lr_pose)  if args.lr_pose  is not None else float(args.lr)
    opt = torch.optim.Adam(
        [
            {"params": [p for p in depth_net.parameters() if p.requires_grad], "lr": lr_depth},
            {"params": [p for p in pose_net.parameters() if p.requires_grad],  "lr": lr_pose},
        ],
        weight_decay=float(args.weight_decay),
    )

    warp_eps = float(args.warp_eps)


    def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
        """t: (B,3,H,W) or (3,H,W), range [0,1]. Returns RGB uint8 HxWx3."""
        if t.ndim == 4:
            t = t[0]
        t = t.detach().float().clamp(0, 1).cpu()
        return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    def _to_uint8_gray(t: torch.Tensor, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
        """t: (B,1,H,W) or (1,H,W) or (H,W). Returns uint8 HxW."""
        if t.ndim == 4:
            t = t[0, 0]
        elif t.ndim == 3:
            t = t[0]
        t = t.detach().float().cpu()
        t = (t - vmin) / max(1e-6, (vmax - vmin))
        t = t.clamp(0, 1)
        return (t.numpy() * 255.0).round().astype(np.uint8)

    def _save_depth_png16(path_out: str, d_m: torch.Tensor, max_depth_m: float):
        """Save depth in millimeters as uint16 PNG."""
        if d_m.ndim == 4:
            d_m = d_m[0, 0]
        elif d_m.ndim == 3:
            d_m = d_m[0]
        d = d_m.detach().float().cpu().clamp(0.0, float(max_depth_m)).numpy()
        d_mm = (d * 1000.0).round().astype(np.uint16)
        cv2.imwrite(path_out, d_mm)

    def _overlay_mask_rgb(rgb: np.ndarray, mask01: np.ndarray, alpha: float) -> np.ndarray:
        """Overlay red mask on RGB image."""
        m = mask01.astype(np.float32)[..., None].clip(0, 1)
        red = np.zeros_like(rgb, dtype=np.float32)
        red[..., 0] = 255.0
        out = rgb.astype(np.float32) * (1 - alpha * m) + red * (alpha * m)
        return out.clip(0, 255).astype(np.uint8)

    def _bbox_from_mask(mask: np.ndarray, thr: float = 0.5):
        ys, xs = np.where(mask > thr)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def save_vis(ep: int, split_name: str, bi: int,
                 rgb0: torch.Tensor, rgb1: torch.Tensor,
                 rgb1_w: torch.Tensor, rgb0_w: torch.Tensor,
                 d0_raw: torch.Tensor, d1_raw: torch.Tensor,
                 d0_ref: torch.Tensor, d1_ref: torch.Tensor,
                 roi0: torch.Tensor, roi1: torch.Tensor,
                 m01: torch.Tensor, m10: torch.Tensor,
                 photo_01: torch.Tensor, photo_10: torch.Tensor,
                 geo01: torch.Tensor, geo10: torch.Tensor):
        """Save a compact set of visuals for paper/debug."""
        base = os.path.join(str(args.out_dir), "vis", f"ep{ep:03d}", split_name)
        os.makedirs(base, exist_ok=True)

        rgb0_np = _to_uint8_rgb(rgb0)
        rgb1_np = _to_uint8_rgb(rgb1)
        rgb1w_np = _to_uint8_rgb(rgb1_w)
        rgb0w_np = _to_uint8_rgb(rgb0_w)

        r0 = roi0.detach().float().cpu()
        r1 = roi1.detach().float().cpu()
        if r0.ndim == 4:
            r0 = r0[0, 0]
        elif r0.ndim == 3:
            r0 = r0[0]
        if r1.ndim == 4:
            r1 = r1[0, 0]
        elif r1.ndim == 3:
            r1 = r1[0]
        roi0_np = r0.numpy()
        roi1_np = r1.numpy()

        thr_obj = float(getattr(args, "yolo_bg_alpha", 0.10)) + 1e-6
        obj0 = (roi0_np > thr_obj).astype(np.float32)
        obj1 = (roi1_np > thr_obj).astype(np.float32)

        over0 = _overlay_mask_rgb(rgb0_np, obj0, float(args.vis_overlay_alpha))
        over1 = _overlay_mask_rgb(rgb1_np, obj1, float(args.vis_overlay_alpha))

        tag = f"b{bi:04d}"

        # RGB + warp
        cv2.imwrite(os.path.join(base, f"{tag}_rgb0.png"), cv2.cvtColor(rgb0_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(base, f"{tag}_rgb1.png"), cv2.cvtColor(rgb1_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(base, f"{tag}_rgb1warp_to0.png"), cv2.cvtColor(rgb1w_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(base, f"{tag}_rgb0warp_to1.png"), cv2.cvtColor(rgb0w_np, cv2.COLOR_RGB2BGR))

        # YOLO ROI overlay
        cv2.imwrite(os.path.join(base, f"{tag}_yolo0_overlay.png"), cv2.cvtColor(over0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(base, f"{tag}_yolo1_overlay.png"), cv2.cvtColor(over1, cv2.COLOR_RGB2BGR))

        # masks & error maps (grayscale)
        cv2.imwrite(os.path.join(base, f"{tag}_mask01.png"), _to_uint8_gray(m01, 0.0, 1.0))
        cv2.imwrite(os.path.join(base, f"{tag}_mask10.png"), _to_uint8_gray(m10, 0.0, 1.0))
        cv2.imwrite(os.path.join(base, f"{tag}_photo01.png"), _to_uint8_gray(photo_01, 0.0, float(args.photo_vis_max)))
        cv2.imwrite(os.path.join(base, f"{tag}_photo10.png"), _to_uint8_gray(photo_10, 0.0, float(args.photo_vis_max)))
        cv2.imwrite(os.path.join(base, f"{tag}_geo01.png"), _to_uint8_gray(geo01, 0.0, float(args.geo_vis_max)))
        cv2.imwrite(os.path.join(base, f"{tag}_geo10.png"), _to_uint8_gray(geo10, 0.0, float(args.geo_vis_max)))

        # depth (mm uint16)
        _save_depth_png16(os.path.join(base, f"{tag}_d0_raw_mm.png"), d0_raw, float(args.max_depth))
        _save_depth_png16(os.path.join(base, f"{tag}_d1_raw_mm.png"), d1_raw, float(args.max_depth))
        _save_depth_png16(os.path.join(base, f"{tag}_d0_ref_mm.png"), d0_ref, float(args.max_depth))
        _save_depth_png16(os.path.join(base, f"{tag}_d1_ref_mm.png"), d1_ref, float(args.max_depth))

        # crop around object region (frame0)
        bb0 = _bbox_from_mask(obj0, thr=0.5)
        if bb0 is not None:
            x0, y0, x1, y1 = bb0
            pad = 10
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(rgb0_np.shape[1] - 1, x1 + pad); y1 = min(rgb0_np.shape[0] - 1, y1 + pad)
            crop_rgb0 = rgb0_np[y0:y1 + 1, x0:x1 + 1]
            crop_over0 = over0[y0:y1 + 1, x0:x1 + 1]
            cv2.imwrite(os.path.join(base, f"{tag}_crop_rgb0.png"), cv2.cvtColor(crop_rgb0, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(base, f"{tag}_crop_yolo0_overlay.png"), cv2.cvtColor(crop_over0, cv2.COLOR_RGB2BGR))

        if bool(args.vis_save_npz):
            np.savez_compressed(
                os.path.join(base, f"{tag}_data.npz"),
                rgb0=rgb0_np, rgb1=rgb1_np,
                rgb1warp_to0=rgb1w_np, rgb0warp_to1=rgb0w_np,
                roi0=roi0_np.astype(np.float32), roi1=roi1_np.astype(np.float32),
                m01=m01.detach().cpu().numpy(), m10=m10.detach().cpu().numpy(),
                photo01=photo_01.detach().cpu().numpy(), photo10=photo_10.detach().cpu().numpy(),
                geo01=geo01.detach().cpu().numpy(), geo10=geo10.detach().cpu().numpy(),
                d0_raw=d0_raw.detach().cpu().numpy(), d1_raw=d1_raw.detach().cpu().numpy(),
                d0_ref=d0_ref.detach().cpu().numpy(), d1_ref=d1_ref.detach().cpu().numpy(),
            )
    def run_loader(dl, train: bool, ep: int, split_name: str):
        if train:
            depth_net.train()
            pose_net.train()
            grad_ctx = torch.enable_grad()
        else:
            depth_net.eval()
            pose_net.eval()
            grad_ctx = torch.no_grad()

        meter = {"total": 0.0, "photo": 0.0, "geo": 0.0, "warp": 0.0, "meas": 0.0, "smooth": 0.0, "benefit": 0.0, "mass": 0.0, "edgetv": 0.0, "scale": 0.0, "coh": 0.0, "reg": 0.0}
        n_batches = 0
        saved_vis = 0
        with grad_ctx:
            it = tqdm(dl, desc=("train" if train else "val"), leave=False, dynamic_ncols=True)
            use_so_now = (int(getattr(args, "so_warmup_epochs", 0)) <= 0) or (ep > int(args.so_warmup_epochs))
            for bi, batch in enumerate(it, 1):
                (x0, x1, d0_raw, d1_raw, vm0, vm1, imu_feat, dtheta, roi0, roi1) = batch
                x0 = x0.to(device, non_blocking=True)
                x1 = x1.to(device, non_blocking=True)
                d0_raw = d0_raw.to(device, non_blocking=True)
                d1_raw = d1_raw.to(device, non_blocking=True)
                vm0 = vm0.to(device, non_blocking=True)
                vm1 = vm1.to(device, non_blocking=True)
                # --- depth roles (A/B/C): measurement mask vm*_meas, dense base d*_base, refined output d*_ref ---
                vm0_meas = vm0
                vm1_meas = vm1

                # When --use_sd_fill is on, x includes a nearest-filled depth channel (dense hint).
                # Build a dense base so holes don't start from 0 during warping/backprojection.
                if bool(getattr(args, "use_sd_fill", False)) and (x0.shape[1] >= 6):
                    d0_fill = x0[:, -1:, :, :]
                    d1_fill = x1[:, -1:, :, :]
                else:
                    d0_fill = d0_raw
                    d1_fill = d1_raw

                d0_base = d0_raw * vm0_meas + d0_fill * (1.0 - vm0_meas)
                d1_base = d1_raw * vm1_meas + d1_fill * (1.0 - vm1_meas)

                d0_base = d0_base.clamp(min=0.0, max=float(args.max_depth))
                d1_base = d1_base.clamp(min=0.0, max=float(args.max_depth))

                imu_feat = imu_feat.to(device, non_blocking=True)
                dtheta = dtheta.to(device, non_blocking=True)
                roi0 = roi0.to(device, non_blocking=True)
                roi1 = roi1.to(device, non_blocking=True)

                rgb0 = x0[:, :3]
                rgb1 = x1[:, :3]

                # Network forward under autocast to reduce VRAM
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Benefit supervision needs a baseline pass with SOFA/BFS disabled.
                    do_benefit = (
                        train
                        and use_so_now
                        and (float(getattr(args, "w_benefit", 0.0)) > 0.0)
                        and (int(getattr(args, "benefit_every", 0)) > 0)
                        and ((bi % int(args.benefit_every)) == 0)
                    )
                    if do_benefit:
                        with torch.no_grad():
                            out0_base, _logvar0_base, _aux0_base = depth_net(x0, use_so=False)

                    out0, _logvar0, _aux0 = depth_net(x0, use_so=use_so_now)
                    out1, _logvar1, _aux1 = depth_net(x1, use_so=use_so_now)
                # Keep geometry/loss in float32 for stability
                out0 = out0.float()
                out1 = out1.float()
                if do_benefit:
                    out0_base = out0_base.float()

                if bool(args.residual_mode):
                    res0 = torch.tanh(out0 * float(args.res_alpha)) * float(args.delta_max)
                    res1 = torch.tanh(out1 * float(args.res_alpha)) * float(args.delta_max)

                    # zero-mean residual per-sample (prevents global DC shift)
                    valid0 = (d0_base > warp_eps).float()
                    den0 = valid0.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
                    mean0 = (res0 * valid0).sum(dim=(2, 3), keepdim=True) / den0
                    res0 = (res0 - mean0).clamp(min=-float(args.delta_max), max=float(args.delta_max))

                    valid1 = (d1_base > warp_eps).float()
                    den1 = valid1.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
                    mean1 = (res1 * valid1).sum(dim=(2, 3), keepdim=True) / den1
                    res1 = (res1 - mean1).clamp(min=-float(args.delta_max), max=float(args.delta_max))

                    # residual is applied on dense base (not raw), and we do NOT zero by vm
                    d0_ref = (d0_base + res0).clamp(min=0.0, max=float(args.max_depth))
                    d1_ref = (d1_base + res1).clamp(min=0.0, max=float(args.max_depth))
                else:
                    # direct prediction; allow non-zero in holes (do NOT multiply by vm)
                    d0_ref = out0.clamp(min=0.0, max=float(args.max_depth))
                    d1_ref = out1.clamp(min=0.0, max=float(args.max_depth))

                # Build baseline refined depth (SOFA/BFS off) for benefit supervision.
                d0_ref_base = None
                if do_benefit:
                    if bool(args.residual_mode):
                        res0b = torch.tanh(out0_base * float(args.res_alpha)) * float(args.delta_max)
                        # zero-mean residual (same as main pass)
                        valid0b = (d0_base > warp_eps).float()
                        den0b = valid0b.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
                        mean0b = (res0b * valid0b).sum(dim=(2, 3), keepdim=True) / den0b
                        res0b = (res0b - mean0b).clamp(min=-float(args.delta_max), max=float(args.delta_max))
                        d0_ref_base = (d0_base + res0b).clamp(min=0.0, max=float(args.max_depth))
                    else:
                        d0_ref_base = out0_base.clamp(min=0.0, max=float(args.max_depth))

                # For self-supervised losses/warping we only require positive depth, not "measured" validity.
                vm0 = (d0_ref > warp_eps).float()
                vm1 = (d1_ref > warp_eps).float()

                # Depth used for projection/warping must be strictly positive
                d0_warp = d0_ref.clamp_min(warp_eps)
                d1_warp = d1_ref.clamp_min(warp_eps)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    t_01, r_res = pose_net(rgb0, rgb1, imu_feat)
                t_01 = t_01.float()
                r_res = r_res.float()
                if bool(args.pose_imu_rot):
                    R_imu = so3_exp(dtheta)
                else:
                    R_imu = torch.eye(3, device=device, dtype=dtheta.dtype).unsqueeze(0).repeat(dtheta.shape[0], 1, 1)
                R_res = so3_exp(r_res)
                R_01 = R_res @ R_imu
                R_10 = R_01.transpose(1, 2)
                t_10 = -(R_10 @ t_01.view(-1, 3, 1)).view(-1, 3)

                                # warp 1->0 (target: frame0, source: frame1)
                rgb1_w, in01, z1_from0 = warp_src_to_tgt(rgb1, d0_warp, R_01, t_01, K)
                d1w, in01_d, _ = warp_src_to_tgt(d1_warp, d0_warp, R_01, t_01, K)
                vm1_w, _, _ = warp_src_to_tgt(vm1.float(), d0_warp, R_01, t_01, K)
                vm1_w = (vm1_w > 0.5).float()

                # Valid mask aligned to target grid (match eval warp-consistency definition)
                m01 = in01 * in01_d * vm0 * vm1_w
                m01 = m01 * (d0_ref > 1e-6).float() * (d1w > 1e-6).float() * (z1_from0 > 1e-6).float()
                if float(args.max_depth) > 0:
                    md = float(args.max_depth)
                    m01 = m01 * (d0_ref < md).float() * (d1w < md).float() * (z1_from0 < md).float()

                occ01 = (z1_from0 <= (d1w + float(args.occ_thresh))).float()
                m01 = m01 * occ01
                # apply YOLO ROI (target frame 0)
                m01 = m01 * roi0
                m01_geom = m01  # keep a copy BEFORE auto-mask (for warp-consistency)

                l1_01 = charbonnier(rgb0 - rgb1_w).mean(dim=1, keepdim=True)
                ssim_01 = ssim(rgb0, rgb1_w)
                photo_01 = float(args.w_ssim) * ssim_01 + (1 - float(args.w_ssim)) * l1_01

                # warp 0->1 (target: frame1, source: frame0)
                rgb0_w, in10, z0_from1 = warp_src_to_tgt(rgb0, d1_warp, R_10, t_10, K)
                d0w, in10_d, _ = warp_src_to_tgt(d0_warp, d1_warp, R_10, t_10, K)
                vm0_w, _, _ = warp_src_to_tgt(vm0.float(), d1_warp, R_10, t_10, K)
                vm0_w = (vm0_w > 0.5).float()

                m10 = in10 * in10_d * vm1 * vm0_w
                m10 = m10 * (d1_ref > 1e-6).float() * (d0w > 1e-6).float() * (z0_from1 > 1e-6).float()
                if float(args.max_depth) > 0:
                    md = float(args.max_depth)
                    m10 = m10 * (d1_ref < md).float() * (d0w < md).float() * (z0_from1 < md).float()

                occ10 = (z0_from1 <= (d0w + float(args.occ_thresh))).float()
                m10 = m10 * occ10
                # apply YOLO ROI (target frame 1)
                m10 = m10 * roi1
                m10_geom = m10  # keep a copy BEFORE auto-mask (for warp-consistency)

                l1_10 = charbonnier(rgb1 - rgb0_w).mean(dim=1, keepdim=True)
                ssim_10 = ssim(rgb1, rgb0_w)
                photo_10 = float(args.w_ssim) * ssim_10 + (1 - float(args.w_ssim)) * l1_10
                if bool(args.auto_mask):
                    id01 = charbonnier(rgb0 - rgb1).mean(dim=1, keepdim=True)
                    id10 = charbonnier(rgb1 - rgb0).mean(dim=1, keepdim=True)
                    m01 = m01 * (photo_01 < id01).float()
                    m10 = m10 * (photo_10 < id10).float()

                photo_loss = (photo_01 * m01).sum() / (m01.sum() + 1e-6) + (photo_10 * m10).sum() / (m10.sum() + 1e-6)

                geo01 = charbonnier(z1_from0 - d1w)
                geo10 = charbonnier(z0_from1 - d0w)
                
                # v10: make geo loss more RGB-reliable by weighting with photo confidence.
                # Intuition: where RGB warp matches well, geometry residual is trusted more.
                # Detach to avoid degenerate solutions through the weighting path.
                wgeo01 = torch.exp(-float(args.geo_photo_beta) * photo_01.detach())
                wgeo10 = torch.exp(-float(args.geo_photo_beta) * photo_10.detach())
                
                geo_loss = (geo01 * m01 * wgeo01).sum() / ((m01 * wgeo01).sum() + 1e-6) + (geo10 * m10 * wgeo10).sum() / ((m10 * wgeo10).sum() + 1e-6)
                
                # v10: geometry-only warp consistency (no RGB mask, no auto-mask).
                # This term should depend primarily on depth geometry + pose.
                if float(args.w_warp) > 0:
                    mw01 = m01_geom  # geometry-only mask (no auto-mask), aligned to target grid
                    mw10 = m10_geom
                    warp01 = charbonnier(z1_from0 - d1w)
                    warp10 = charbonnier(z0_from1 - d0w)
                    warp_loss = (warp01 * mw01).sum() / (mw01.sum() + 1e-6) + (warp10 * mw10).sum() / (mw10.sum() + 1e-6)
                else:
                    warp_loss = torch.zeros_like(photo_loss)
                # v10: measurement depth supervision with per-batch alignment (scale / affine).
                # This stabilizes training and avoids the supervision being dominated by global bias/scale.
                if args.meas_align == "none":
                    d0_al = d0_ref
                    d1_al = d1_ref
                    a0 = b0 = a1 = b1 = None
                    # Optional: sparse measurement supervision (prevents L_meas from forcing ref≈raw everywhere)
                    if bool(getattr(args, "meas_sparse", False)):
                        kp = float(args.meas_keep_prob) if (args.meas_keep_prob is not None) else float(args.obs_keep_prob)
                        mf = float(args.meas_max_frac) if (args.meas_max_frac is not None) else float(args.obs_max_frac)

                        def _sparse_mask(vm: torch.Tensor) -> torch.Tensor:
                            m = (vm > 0.5)
                            frac = float(m.float().mean().item())
                            if (kp < 1.0) and (frac > mf):
                                keep = (torch.rand_like(vm) < kp)
                                m = m & keep
                            return m.float()

                        vm0_meas = _sparse_mask(vm0_meas)
                        vm1_meas = _sparse_mask(vm1_meas)
                    else:
                        vm0_meas = vm0_meas
                        vm1_meas = vm1_meas

                    meas_loss = meas_residual(d0_al, d0_raw, vm0_meas, args) + meas_residual(d1_al, d1_raw, vm1_meas, args)
                    reg_align = torch.zeros_like(meas_loss)
                elif args.meas_align == "scale":
                    a0 = fit_scale_only(d0_ref, d0_raw, vm0_meas)
                    a1 = fit_scale_only(d1_ref, d1_raw, vm1_meas)
                    d0_al = a0 * d0_ref
                    d1_al = a1 * d1_ref
                    meas_loss = meas_residual(d0_al, d0_raw, vm0_meas, args) + meas_residual(d1_al, d1_raw, vm1_meas, args)
                    reg_align = float(args.w_scale_reg) * (((a0 - 1.0) ** 2).mean() + ((a1 - 1.0) ** 2).mean())
                else:
                    a0, b0 = fit_affine_scale_shift(d0_ref, d0_raw, vm0_meas)
                    a1, b1 = fit_affine_scale_shift(d1_ref, d1_raw, vm1_meas)
                    d0_al = a0 * d0_ref + b0
                    d1_al = a1 * d1_ref + b1
                    meas_loss = meas_residual(d0_al, d0_raw, vm0_meas, args) + meas_residual(d1_al, d1_raw, vm1_meas, args)
                    # Regularize (a->1, b->0) to remove constant DoF and suppress global bias long-term.
                    reg_align = float(args.w_scale_reg) * (((a0 - 1.0) ** 2).mean() + ((a1 - 1.0) ** 2).mean())                               + float(args.w_shift_reg) * ((b0 ** 2).mean() + (b1 ** 2).mean())

                smooth_loss = edge_aware_smoothness(d0_ref, rgb0, vm0) + edge_aware_smoothness(d1_ref, rgb1, vm1)

                # -------------------------
                # BFS benefit supervision (measurement residual improvement)
                # NOTE: target is built with detached tensors to avoid meta-learning collapse.
                # -------------------------
                benefit_loss = d0_ref.new_zeros(())
                if do_benefit and (d0_ref_base is not None):
                    with torch.no_grad():
                        # Align baseline depth to measurement the same way as main meas loss does
                        if args.meas_align == "none":
                            d0_base_al = d0_ref_base
                        elif args.meas_align == "scale":
                            a0b = fit_scale_only(d0_ref_base, d0_raw, vm0_meas)
                            d0_base_al = a0b * d0_ref_base
                        else:
                            a0b, b0b = fit_affine_scale_shift(d0_ref_base, d0_raw, vm0_meas)
                            d0_base_al = a0b * d0_ref_base + b0b

                        # Use the already aligned main prediction but DETACH it
                        d0_fov_al_det = d0_al.detach()
                        # Build benefit target (measurement-only or measurement+photo)
                        if bool(getattr(args, "benefit_use_photo", False)):
                            # Photometric component: supervise also on measurement holes where warping is valid.
                            # Compute baseline photometric error map with no_grad (target is detached anyway).
                            with torch.no_grad():
                                d0_base_warp = d0_ref_base.clamp_min(warp_eps)
                                rgb1_w_base, in01_base, _ = warp_src_to_tgt(rgb1, d0_base_warp, R_01, t_01, K)
                                l1_01_base = charbonnier(rgb0 - rgb1_w_base).mean(dim=1, keepdim=True)
                                ssim_01_base = ssim(rgb0, rgb1_w_base)
                                photo_01_base = float(args.w_ssim) * ssim_01_base + (1 - float(args.w_ssim)) * l1_01_base

                            # Use foveated photometric error map from the main forward, but DETACH it.
                            photo_01_fov_det = photo_01.detach()

                            # Warp-valid mask: use main geometry mask (pre-auto-mask) AND baseline in-bounds.
                            photo_mask = (m01_geom.detach() * in01_base.detach())
                            if not bool(getattr(args, "benefit_photo_on_all", False)):
                                photo_mask = photo_mask * (1.0 - vm0_meas)

                            benefit_tgt = build_benefit_target_from_meas_and_photo(
                                d_base_al=d0_base_al,
                                d_fov_al_det=d0_fov_al_det,
                                d_meas=d0_raw,
                                meas_mask=vm0_meas,
                                photo_base=photo_01_base.detach(),
                                photo_fov_det=photo_01_fov_det,
                                photo_mask=photo_mask,
                                args=args,
                            )
                        else:
                            benefit_tgt = build_benefit_target_from_meas_residual(
                                d_base_al=d0_base_al,
                                d_fov_al_det=d0_fov_al_det,
                                d_meas=d0_raw,
                                meas_mask=vm0_meas,
                                args=args,
                            )

                    bce = F.binary_cross_entropy_with_logits(_aux0["bfs_logits"], benefit_tgt, reduction="none")
                    denom = None
                    if bool(getattr(args, "benefit_use_photo", False)):
                        # union of measurement-valid and photo-valid pixels
                        um = ((vm0_meas > 0.5) | (photo_mask > 0.5)).float()
                        denom = um.sum().clamp_min(1.0)
                        benefit_loss = (bce * um).sum() / denom
                    else:
                        denom = vm0_meas.sum().clamp_min(1.0)
                        benefit_loss = (bce * vm0_meas).sum() / denom

                # -------------------------
                # BFS mask regularizers (do not require GT)
                # -------------------------
                loss_mass = d0_ref.new_zeros(())
                loss_edgetv = d0_ref.new_zeros(())
                loss_scale = d0_ref.new_zeros(())
                loss_coh = d0_ref.new_zeros(())
                if use_so_now and isinstance(_aux0, dict) and ("m" in _aux0):
                    mp = _aux0["m"]
                    loss_mass = (mp.mean() - float(args.bfs_p)) ** 2
                    loss_edgetv = edge_aware_tv(mp, rgb0)
                    # encourage using small-object scale branch (same sign as v4)
                    loss_scale = -_aux0["w_small"].mean()
                    if float(getattr(args, "w_mask_coh", 0.0)) > 0:
                        loss_coh = mask_local_coherence_loss(mp, rgb0, ks=int(getattr(args, "coh_ks", 7)))


                total = float(args.w_photo) * photo_loss + float(args.w_geo) * geo_loss + float(args.w_warp) * warp_loss + float(args.w_meas) * meas_loss + float(args.w_smooth) * smooth_loss + reg_align + float(getattr(args, 'w_benefit', 0.0)) * benefit_loss + float(getattr(args, 'w_mass', 0.0)) * loss_mass + float(getattr(args, 'w_edgetv', 0.0)) * loss_edgetv + float(getattr(args, 'w_scale_bfs', 0.0)) * loss_scale + float(getattr(args, 'w_mask_coh', 0.0)) * loss_coh

                # visualization (optional, for paper/debug)
                if int(args.vis_every) > 0 and saved_vis < int(args.vis_max_per_epoch):
                    want = (args.vis_split == "both") or (train and args.vis_split == "train") or ((not train) and args.vis_split == "val")
                    if want and ((bi - 1) % int(args.vis_every) == 0):
                        try:
                            save_vis(ep, split_name, bi,
                                     rgb0, rgb1, rgb1_w, rgb0_w,
                                     d0_raw, d1_raw, d0_ref, d1_ref,
                                     roi0, roi1, m01, m10,
                                     photo_01, photo_10, geo01, geo10)
                            saved_vis += 1
                        except Exception:
                            pass



                # progress display
                try:
                    it.set_postfix(total=float(total.item()), photo=float(photo_loss.item()), geo=float(geo_loss.item()), warp=float(warp_loss.item()), meas=float(meas_loss.item()), benefit=float(benefit_loss.item()))
                except Exception:
                    pass
                if train:
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(total).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        total.backward()
                        opt.step()

                meter["total"] += float(total.item())
                meter["photo"] += float(photo_loss.item())
                meter["geo"] += float(geo_loss.item())
                meter["warp"] += float(warp_loss.item())
                meter["meas"] += float(meas_loss.item())
                meter["smooth"] += float(smooth_loss.item())
                meter["benefit"] += float(benefit_loss.item())
                meter["mass"] += float(loss_mass.item())
                meter["edgetv"] += float(loss_edgetv.item())
                meter["scale"] += float(loss_scale.item())
                meter["coh"] += float(loss_coh.item())
                meter["reg"] += float(reg_align.item())
                n_batches += 1

        for k in meter:
            meter[k] /= max(1, n_batches)
        return meter

    best_metric = float("inf")

    for ep in range(1, int(args.epochs) + 1):
        print(f"\n=== Epoch {ep}/{int(args.epochs)} ===")
        tr = run_loader(dl_tr, train=True, ep=ep, split_name="train")
        if dl_va is not None:
            va = run_loader(dl_va, train=False, ep=ep, split_name="val")
            metric = va["total"]
            print(f"[epoch {ep}] train: total={tr['total']:.6f} photo={tr['photo']:.6f} geo={tr['geo']:.6f} warp={tr['warp']:.6f} meas={tr['meas']:.6f} smooth={tr['smooth']:.6f} reg={tr['reg']:.6f}")
            print(f"[epoch {ep}]   val: total={va['total']:.6f} photo={va['photo']:.6f} geo={va['geo']:.6f} warp={va['warp']:.6f} meas={va['meas']:.6f} smooth={va['smooth']:.6f} reg={va['reg']:.6f}")
        else:
            va = None
            metric = tr["total"]
            print(f"[epoch {ep}] train: total={tr['total']:.6f} photo={tr['photo']:.6f} geo={tr['geo']:.6f} warp={tr['warp']:.6f} meas={tr['meas']:.6f} smooth={tr['smooth']:.6f} reg={tr['reg']:.6f}")

        # save last (overwrite)
        ckpt = {
            "epoch": ep,
            "model": depth_net.state_dict(),
            "pose": pose_net.state_dict(),
            "opt": opt.state_dict(),
            "K": {"fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy},
            "imu_R_g2c": ds_full.R_g2c.tolist(),
            "args": vars(args),
            "train": tr,
            "val": va,
        }
        last_path = os.path.join(args.out_dir, "ckpt_last.pth")
        torch.save(ckpt, last_path)

        if metric < best_metric:
            best_metric = metric
            best_path = os.path.join(args.out_dir, "ckpt_best.pth")
            torch.save(ckpt, best_path)
            print(f"[save] best -> {best_path} (metric={best_metric:.6f})")

        if bool(args.save_all):
            ep_path = os.path.join(args.out_dir, f"ckpt_ep{ep:03d}.pth")
            torch.save(ckpt, ep_path)
            print(f"[save] {ep_path}")

    print(f"[done] best_metric={best_metric:.6f} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()