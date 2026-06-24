# -*- coding: utf-8 -*-
"""
Official training script for the public VOID benchmark experiment in the RG-RGD paper.

This script trains/evaluates the supervised RGB-D depth refinement model with the
BFS-SOFA small-target focusing module on VOID. Optional pseudo-mask distillation
flags are retained for ablation but are disabled in the default paper-reproduction command.

Example:
    python tools/train_void_supervised.py \
        --root /path/to/void_release/void_1500 \
        --out_dir runs/void_rgrgd \
        --void_split official \
        --epochs 30 \
        --img_h 480 --img_w 640 \
        --batch_size 6 \
        --lr 2.8e-4 \
        --lr_vit 1.0e-5 \
        --vit_no_pretrained \
        --ema_enable --ema_decay 0.9999 --ema_eval

Notes:
    - Replace dataset/model paths with local paths before running.
    - Optional local ViT weights can be provided with --vit_local_weights.
    - The script uses public datasets only; no private data are included.
"""

import os, glob, random, argparse, time, math, copy
from typing import List, Tuple, Dict, Optional

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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Optional deps
# -------------------------
try:
    import timm
except Exception:
    timm = None

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
                print(f"[warn] timm ViT create_model failed ({self.model_name}): {e}. Falling back to CNN stem.")
                self.vit = None

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
    Missing file -> None (caller should handle).
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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

    # If no exact density directory is found, return the first candidate for diagnostics.
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
        vm = (vm_u16 > 0).astype(np.float32)

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

    def forward(self, f_q: torch.Tensor, f_kv: torch.Tensor, mask_so: torch.Tensor):
        """Cross attention from query features f_q to key/value features f_kv,
        with a soft attention bias derived from mask_so.
        """
        B, C, H, W = f_q.shape
        tq = self.pe_q(f_q)     # (B, Nq, C)
        tkv = self.pe_kv(f_kv)  # (B, Nk, C)

        # Downsample mask to token grid and flatten to (B, Nk)
        mk = F.interpolate(mask_so, size=(H // self.patch, W // self.patch), mode="nearest")
        mk = mk.flatten(2).transpose(1, 2).squeeze(-1)  # (B, Nk)

        # Attention bias: suppress non-small-object tokens for KV
        bias = -self.bias_scale * (1.0 - mk).clamp(0, 1)          # (B, Nk)
        bias = bias.unsqueeze(1).repeat(1, tq.shape[1], 1)        # (B, Nq, Nk)
        bias = bias.repeat_interleave(self.heads, dim=0)          # (B*heads, Nq, Nk)

        attn_out, _ = self.attn(tq, tkv, tkv, attn_mask=bias, need_weights=False)
        tq = self.norm1(tq + attn_out)
        tq = self.norm2(tq + self.ff(tq))
        out = self.un(tq, H, W)
        return out

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
        global YOLO
        if YOLO is None:
            try:
                from ultralytics import YOLO as _YOLO
            except Exception as e:
                raise ImportError("ultralytics is required for YOLO. Install with: pip install -r requirements-optional.txt") from e
            YOLO = _YOLO
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
    - LiteLearnedPropRefiner provides the lightweight propagation step.
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
# Full RG-RGD network
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
    ):
        super().__init__()
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

        # proxy params (kept for compatibility, but BFS replaces much of it)
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

    def forward(self, x, use_so: bool = True):
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
            f_rgb = f_rgb + self.sofa(f_rgb, f_dep, mask_so)
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

        # anchor observed/fused measurements before refiners (used to keep sparse samples stable)
        obs = (vm > 0.5) & (depth > 1e-6)
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
        return mu_ref, logvar_ref, aux


# -------------------------
# Loss + distillation + NLL warmup driven by crit.w_nll updated per-epoch
# -------------------------
class RGRGDDepthLoss(nn.Module):
    def __init__(
        self,
        w_smooth=0.01,
        w_so=1.0,
        w_mask_sparse=1e-4,
        w_calib=0.0,
        so_gamma=2.0,
        w_l1=1.0,
        w_sparse=0.1,
        w_sigma_obs=0.02,
        w_nll=0.0,                    # will be scheduled (warmup)
        sparse_agree_tau=0.2,
        grad_clip_sigma=16.0,
        min_sigma: float = 0.02,
        max_sigma: float = 10.0,
        # distillation
        w_mask_distill=0.2,
        teacher_thr=0.5,
        # BFS extras
        bfs_p=0.05,
        w_mass=0.1,
        w_edgetv=0.01,
        w_scale=0.1,
        # detail & mask stability
        w_grad=0.2,
        grad_edge_k=2.0,
        w_mask_coh=0.05,
        coh_ks=7,
    ):
        super().__init__()
        self.w_smooth = float(w_smooth)
        self.w_so = float(w_so)
        self.w_mask_sparse = float(w_mask_sparse)
        self.w_calib = float(w_calib)
        self.so_gamma = float(so_gamma)
        self.w_l1 = float(w_l1)
        self.w_sparse = float(w_sparse)
        self.w_sigma_obs = float(w_sigma_obs)
        self.w_nll = float(w_nll)
        self.sparse_agree_tau = float(sparse_agree_tau)
        self.grad_clip_sigma = float(grad_clip_sigma)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.min_logvar = float(math.log(self.min_sigma ** 2 + 1e-12))
        self.max_logvar = float(math.log(self.max_sigma ** 2 + 1e-12))

        self.w_mask_distill = float(w_mask_distill)
        self.teacher_thr = float(teacher_thr)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

        # BFS
        self.bfs_p = float(bfs_p)
        self.w_mass = float(w_mass)
        self.w_edgetv = float(w_edgetv)
        self.w_scale = float(w_scale)
        self.w_grad = float(w_grad)
        self.grad_edge_k = float(grad_edge_k)
        self.w_mask_coh = float(w_mask_coh)
        self.coh_ks = int(coh_ks)


    def forward(self, x_in, mu, logvar, gt, gt_valid, aux: dict, teacher_mask: Optional[torch.Tensor] = None):
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        sigma = torch.exp(0.5 * logvar)
        e2 = (mu - gt) ** 2
        inv_var = torch.exp(-logvar)
        nll_pix = 0.5 * inv_var * e2 + 0.5 * logvar
        nll = (nll_pix * gt_valid).sum() / (gt_valid.sum() + 1e-6)

        mask_so = aux["mask_so"].detach()
        w = (1.0 + self.w_so * (mask_so ** self.so_gamma)).clamp(1.0, 1.0 + self.w_so)
        nll_so = (nll_pix * gt_valid * w).sum() / ((gt_valid * w).sum() + 1e-6)

        l1 = ((mu - gt).abs() * gt_valid).sum() / (gt_valid.sum() + 1e-6)

        sd = x_in[:, 3:4]
        vm = x_in[:, 4:5]
        obs = ((vm > 0.5) & (sd > 1e-6)).float()
        agree = obs * gt_valid
        if self.sparse_agree_tau and self.sparse_agree_tau > 0:
            agree = agree * ((sd - gt).abs() < self.sparse_agree_tau).float()
        loss_sparse = ((mu - sd).abs() * agree).sum() / (agree.sum() + 1e-6)

        # BMF: measurement NLL to learn sigma_obs (measurement reliability)
        loss_sigma_obs = mu.new_tensor(0.0)
        sigma_obs = aux.get("sigma_obs", None)
        if sigma_obs is not None:
            var_o = (sigma_obs ** 2).clamp(1e-12, 1e6)
            e2_meas = (sd - gt) ** 2
            nll_meas_pix = 0.5 * e2_meas / var_o + 0.5 * torch.log(var_o)
            denom = (obs * gt_valid).sum() + 1e-6
            loss_sigma_obs = (nll_meas_pix * obs * gt_valid).sum() / denom
        else:
            # fallback: mild penalty on predicted sigma at observed pixels
            sigma_c = sigma.clamp(1e-6, self.grad_clip_sigma)
            loss_sigma_obs = (sigma_c * obs).sum() / (obs.sum() + 1e-6)

        loss_smooth = edge_aware_smoothness(mu, x_in[:, :3]) if self.w_smooth > 0 else mu.new_tensor(0.0)
        mp = aux.get("mask_so", aux["m"])  # align budget/TV with SOFA-used mask
        loss_mask_sparse = mp.mean()

        loss_calib = mu.new_tensor(0.0)
        if self.w_calib > 0:
            abs_err = (mu - gt).abs()
            target = (abs_err / (sigma + 1e-6)).clamp(0, 10)
            loss_calib = ((target - 1.0).abs() * gt_valid).sum() / (gt_valid.sum() + 1e-6)

        # ---- Direction-A: teacher mask distillation (train-time only) ----
        loss_mask_distill = mu.new_tensor(0.0)
        if teacher_mask is not None and self.w_mask_distill > 0:
            # teacher_mask: (B,1,H,W) in [0,1]
            t = (teacher_mask > self.teacher_thr).float()
            logits = aux["bfs_logits"]  # now from BFS
            # only supervise where gt_valid exists? (safer) you can also supervise everywhere
            sup = (gt_valid > 0.5).float()
            bce = self.bce_logits(logits, t)
            loss_mask_distill = (bce * sup).sum() / (sup.sum() + 1e-6)

        # ---- BFS extras ----
        loss_mass = (mp.mean() - self.bfs_p) ** 2
        loss_edgetv = edge_aware_tv(mp, x_in[:, :3])
        loss_scale = -aux["w_small"].mean()

        loss_grad = depth_grad_l1(mu, gt, gt_valid, x_in[:, :3], edge_k=self.grad_edge_k) if self.w_grad > 0 else mu.new_tensor(0.0)
        loss_mask_coh = mask_local_coherence_loss(mp, x_in[:, :3], ks=self.coh_ks) if self.w_mask_coh > 0 else mu.new_tensor(0.0)


        total = (
            self.w_nll * nll
            + 0.5 * self.w_nll * nll_so
            + self.w_l1 * l1
            + self.w_sparse * loss_sparse
            + self.w_sigma_obs * loss_sigma_obs
            + self.w_smooth * loss_smooth
            + self.w_mask_sparse * loss_mask_sparse
            + self.w_calib * loss_calib
            + self.w_mask_distill * loss_mask_distill
            # BFS
            + self.w_mass * loss_mass
            + self.w_edgetv * loss_edgetv
            + self.w_scale * loss_scale
            + self.w_grad * loss_grad
            + self.w_mask_coh * loss_mask_coh
        )

        return {
            "loss": total,
            "nll": nll.detach().item(),
            "nll_so": nll_so.detach().item(),
            "l1": l1.detach().item(),
            "sparse": loss_sparse.detach().item(),
            "sigma_obs": loss_sigma_obs.detach().item(),
            "smooth": loss_smooth.detach().item() if torch.is_tensor(loss_smooth) else 0.0,
            "mask_sparse": loss_mask_sparse.detach().item(),
            "mask_distill": loss_mask_distill.detach().item(),
            "mass": loss_mass.detach().item(),
            "edgetv": loss_edgetv.detach().item(),
            "scale": loss_scale.detach().item(),
            "grad": loss_grad.detach().item() if torch.is_tensor(loss_grad) else 0.0,
            "mask_coh": loss_mask_coh.detach().item() if torch.is_tensor(loss_mask_coh) else 0.0,
            "w_nll": float(self.w_nll),
        }


# -------------------------
# Evaluation (+ optional TTA scale/bias)
# -------------------------
class ModelEMA:
    """Exponential Moving Average of model weights.

    In practice, EMA often gives a small but reliable boost on val (especially for MAE),
    with very little extra cost.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_min: float,
    eval_max: float,
    tta_steps: int = 0,
    tta_lr: float = 5e-2,
    teacher_enable: bool = False,
    teacher_thr: float = 0.5,
):
    model.eval()
    mae_sum, rmse_sum, absrel_sum, n_sum = 0.0, 0.0, 0.0, 0
    mae_roi_sum, roi_n = 0.0, 0

    for batch in loader:
        if teacher_enable:
            x, gt, gtv, mteach = batch
            mteach = mteach.to(device, non_blocking=True)
        else:
            x, gt, gtv = batch
            mteach = None

        x = x.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        gtv = gtv.to(device, non_blocking=True)
        valid = apply_eval_range(gt, gtv, eval_min, eval_max)

        with torch.inference_mode():
            mu, logvar, _ = model(x, use_so=True)
  # eval uses SOFA/BFS

        # Optional test-time adaptation to fit (a*mu+b) on sparse depth points
        if tta_steps and tta_steps > 0:
            sd = x[:, 3:4]
            vm = x[:, 4:5]
            obs = ((vm > 0.5) & (sd > 1e-6)).float()

            a = torch.ones(1, device=device, requires_grad=True)
            b = torch.zeros(1, device=device, requires_grad=True)
            opt = torch.optim.SGD([a, b], lr=tta_lr)
            for _ in range(int(tta_steps)):
                opt.zero_grad(set_to_none=True)
                mu_ab = a * mu + b
                loss = ((mu_ab - sd).abs() * obs).sum() / (obs.sum() + 1e-6)
                loss.backward()
                opt.step()
            mu = (a * mu + b).detach()

        m = depth_metrics(mu, gt, valid)
        mae_sum += m["mae_m"]
        rmse_sum += m["rmse_m"]
        absrel_sum += m["absrel"]
        n_sum += 1

        # ROI MAE (if teacher mask is provided)
        if teacher_enable and (mteach is not None):
            roi = (mteach > float(teacher_thr)).float()
            valid_roi = valid * roi
            if valid_roi.sum() > 10.0:
                m_roi = depth_metrics(mu, gt, valid_roi)
                mae_roi_sum += m_roi["mae_m"]
                roi_n += 1

    if n_sum == 0:
        return {"mae_m": 0.0, "rmse_m": 0.0, "absrel": 0.0}

    out = {"mae_m": mae_sum / n_sum, "rmse_m": rmse_sum / n_sum, "absrel": absrel_sum / n_sum}
    if teacher_enable and roi_n > 0:
        out["mae_roi_m"] = mae_roi_sum / roi_n
    return out


# -------------------------
# Train
# -------------------------
# -------------------------
# -------------------------
# Train
# -------------------------
def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Resolve training image size
    if getattr(args, "img_size", None) is not None:
        # legacy square resize
        img_hw = (int(args.img_size), int(args.img_size))
    else:
        img_hw = (int(args.img_h), int(args.img_w))

    density_dir = infer_void_density_dir(args.root, getattr(args, "density", None))

    print("Device:", device)
    print("VOID density dir:", density_dir)
    print("VOID split source:", args.void_split)

    if args.void_split == "official":
        # Official protocol: use official train list for training and official test list for validation.
        tr_pairs = load_void_pairs_from_official_txt(
            density_dir, "train", debug_show=int(args.official_debug_txt)
        )
        val_pairs = load_void_pairs_from_official_txt(
            density_dir, "test", debug_show=0
        )

        # Optional caps for quick local checks.
        # If --max_total > 0, interpret it as a TOTAL budget across train+val, using --val_ratio as the val fraction.
        if args.max_total and args.max_total > 0:
            vr = float(args.val_ratio) if args.val_ratio and args.val_ratio > 0 else 0.0
            n_val = int(round(args.max_total * vr))
            n_val = max(1, n_val) if vr > 0 else 0
            n_tr = max(1, int(args.max_total) - n_val)
        else:
            n_tr = 0
            n_val = 0

        tr_pairs = cap_pairs_by_scene_and_total(
            tr_pairs,
            max_scenes=args.max_scenes,
            per_scene_cap=args.per_scene_cap,
            max_total=n_tr,
            seed=args.seed,
        )
        # For val, keep full official test unless the user capped total and requested a non-zero val fraction.
        val_pairs = cap_pairs_by_scene_and_total(
            val_pairs,
            max_scenes=0,
            per_scene_cap=0,
            max_total=n_val,
            seed=args.seed,
        )

        total_used = len(tr_pairs) + len(val_pairs)
        print(
            f"[official] train={len(tr_pairs)} | test(val)={len(val_pairs)} | total_used={total_used} "
            f"| seed={args.seed} | void_split=official"
        )
    else:
        # Legacy: scan all scenes under data/ (may not match official protocol).
        pairs = find_void_pairs_via_scenes(
            density_dir,
            max_scenes=args.max_scenes,
            max_total=args.max_total,
            verbose=True,
            per_scene_cap=args.per_scene_cap,
            seed=args.seed,
        )
        tr_pairs, val_pairs, info = split_pairs_train_val(pairs, args.val_ratio, args.split_mode, args.seed)
        print(info)
        print(
            f"Total used: {len(pairs)} | train: {len(tr_pairs)} | val: {len(val_pairs)} "
            f"| seed={args.seed} | split={args.split_mode} | void_split={args.void_split}"
        )

    # ---- Dataset ----
    ds_tr = VoidDataset(
        tr_pairs,
        img_size=img_hw,
        depth_scale=args.depth_scale,
        augment=True,
        teacher_enable=args.teacher_enable,
        teacher_subdir=args.teacher_subdir,
        teacher_ext=args.teacher_ext,
        teacher_missing_as_zero=True,
        aug_flip_p=args.aug_flip_p,
        aug_color_p=args.aug_color_p,
        aug_rot_p=args.aug_rot_p,
        aug_rot_deg=args.aug_rot_deg,
        use_sd_fill=not args.no_sd_fill,
        # temporal fusion (hint only)
        temporal_radius=args.temporal_radius,
        temporal_mode=args.temporal_mode,
        # ROI crop (teacher-guided)
        roi_crop_p=args.roi_crop_p,
        roi_crop_scale_min=args.roi_crop_scale_min,
        roi_crop_scale_max=args.roi_crop_scale_max,
        roi_crop_thr=args.roi_crop_thr,
    )
    ds_va = VoidDataset(
        val_pairs,
        img_size=img_hw,
        depth_scale=args.depth_scale,
        augment=False,
        teacher_enable=args.teacher_enable,  # for loader unpack symmetry
        teacher_subdir=args.teacher_subdir,
        teacher_ext=args.teacher_ext,
        teacher_missing_as_zero=True,
        aug_flip_p=0.0,
        aug_color_p=0.0,
        aug_rot_p=0.0,
        aug_rot_deg=0.0,
        use_sd_fill=not args.no_sd_fill,
        temporal_radius=args.temporal_radius,
        temporal_mode=args.temporal_mode,
        roi_crop_p=0.0,
        roi_crop_scale_min=args.roi_crop_scale_min,
        roi_crop_scale_max=args.roi_crop_scale_max,
        roi_crop_thr=args.roi_crop_thr,
    )

    # ---- DataLoader (Windows-friendly): persistent workers + prefetch ----
    _dl_tr_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if int(args.num_workers) > 0:
        _dl_tr_kwargs.update(
            persistent_workers=True,  # keep workers alive (helps Windows)
            prefetch_factor=4,
        )
    dl_tr = DataLoader(ds_tr, **_dl_tr_kwargs)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Model ----
    model = RGRGDDepthRefiner(
        base=args.base,
        patch=args.patch,
        heads=args.heads,
        bp_iters=args.bp_iters,
        sofa_bias_scale=args.sofa_bias_scale,
        proxy_logit_scale=args.proxy_logit_scale,
        learn_kmap=not args.no_kmap,
        vit_name=args.vit_name,
        vit_patch=args.vit_patch,
        vit_pretrained=not args.vit_no_pretrained,
        vit_local_weights=args.vit_local_weights,
        yolo_mode=args.yolo_mode,
        yolo_ckpt=args.yolo_ckpt,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_small_area_max=args.yolo_small_area_max,
        yolo_weight=args.yolo_weight,
        win_size=args.win_size,
        tf_depth=args.tf_depth,
        init_logvar=args.init_logvar,
        bfs_p=args.bfs_p,
        bfs_t=args.bfs_t,
        mask_blur_ks=args.mask_blur_ks,
        mask_blur_sigma=args.mask_blur_sigma,
        so_budget_norm=not args.no_so_budget_norm,
        bp_beta=args.bp_beta,
        bp_refine_var=args.bp_refine_var,
        use_residual=not args.no_residual,
        hard_sparse_copy=not args.no_hard_sparse_copy,
        max_depth=args.max_depth,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        # CSPN-lite refinement
        cspn_enable=not args.no_cspn,
        cspn_iters=args.cspn_iters,
        cspn_hidden=args.cspn_hidden,
        cspn_use_diag=True,
    ).to(device)

    # ---- EMA ----
    ema = None
    if args.ema_enable:
        ema = ModelEMA(model, decay=float(args.ema_decay))

    # ---- Loss ----
    crit = RGRGDDepthLoss(
        w_smooth=args.w_smooth,
        w_so=args.w_so,
        w_mask_sparse=args.w_mask_sparse,
        w_calib=args.w_calib,
        so_gamma=args.so_gamma,
        w_l1=args.w_l1,
        w_sparse=args.w_sparse,
        w_sigma_obs=args.w_sigma_obs,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        w_nll=0.0,  # scheduled
        sparse_agree_tau=args.sparse_agree_tau,
        w_mask_distill=args.w_mask_distill if args.teacher_enable else 0.0,
        teacher_thr=args.teacher_thr,
        # BFS
        bfs_p=args.bfs_p,
        w_mass=args.w_mass,
        w_edgetv=args.w_edgetv,
        w_scale=args.w_scale,
        w_grad=args.w_grad,
        grad_edge_k=args.grad_edge_k,
        w_mask_coh=args.w_mask_coh,
        coh_ks=args.coh_ks,
    )

    # ---- Optimizer ----
    base_lr = float(args.lr)

    vit = getattr(model.rgb_stem, "vit", None)
    vit_params = list(vit.parameters()) if vit is not None else []
    vit_param_ids = {id(p) for p in vit_params}
    other_params = [p for p in model.parameters() if id(p) not in vit_param_ids]

    # Separate LR for ViT if present (helps stability).
    param_groups = [{"params": other_params, "lr": base_lr}]
    if len(vit_params) > 0:
        param_groups.append({"params": vit_params, "lr": float(args.lr_vit)})

    opt = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and device.type == "cuda"))

    best = 1e9
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "rgrgd_void_best.pth")

    for ep in range(1, args.epochs + 1):
        # ---- LR schedule (warmup + cosine) ----
        warm = max(1, int(args.lr_warmup_epochs))
        if ep <= warm:
            lr_scale = ep / warm
            lr_main = base_lr * lr_scale
            lr_vit = float(args.lr_vit) * lr_scale
        else:
            # cosine decay to lr_min_ratio
            t = (ep - warm) / max(1, (args.epochs - warm))
            cos = 0.5 * (1.0 + math.cos(math.pi * float(t)))
            lr_main_min = base_lr * float(args.lr_min_ratio)
            lr_vit_base = float(args.lr_vit)
            lr_vit_min = lr_vit_base * float(args.lr_min_ratio)
            lr_main = lr_main_min + (base_lr - lr_main_min) * cos
            lr_vit = lr_vit_min + (lr_vit_base - lr_vit_min) * cos

        opt.param_groups[0]["lr"] = lr_main
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr_vit

        # ---- optional freeze ViT ----
        if int(args.freeze_vit_epochs) > 0:
            model.rgb_stem.set_vit_trainable(ep > int(args.freeze_vit_epochs))

        # ---- w_nll warmup ----
        wn_warm = max(1, int(args.nll_warmup_epochs))
        wn_scale = min(1.0, ep / wn_warm) if wn_warm > 0 else 1.0
        crit.w_nll = float(args.w_nll_max) * wn_scale

        model.train()
        pbar = tqdm(dl_tr, total=len(dl_tr), ncols=120)
        loss_meter = 0.0
        mae_meter = 0.0
        nll_meter = 0.0
        md_meter = 0.0
        benefit_meter = 0.0

        # Gradient accumulation (effective batch = batch_size * accum_steps)
        accum = max(1, int(getattr(args, "accum_steps", 1)))

        t_ep0 = time.time()

        for it, batch in enumerate(pbar, 1):
            if args.teacher_enable:
                x, gt, gtv, mteach = batch
                mteach = mteach.to(device, non_blocking=True)
            else:
                x, gt, gtv = batch
                mteach = None

            x = x.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            gtv = gtv.to(device, non_blocking=True)

            # optional: keep train masking consistent with eval range
            gtv_use = apply_eval_range(gt, gtv, args.eval_min, args.eval_max) if args.train_apply_eval_range else gtv

            if (it - 1) % accum == 0:
                opt.zero_grad(set_to_none=True)

            # compute benefit supervision only when enabled and on schedule
            use_so_now = (int(args.so_warmup_epochs) <= 0) or (ep > int(args.so_warmup_epochs))
            do_benefit = (
                use_so_now
                and (args.w_benefit > 0)
                and (args.benefit_every > 0)
                and ((it % args.benefit_every) == 0)
            )

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                if do_benefit:
                    with torch.no_grad():
                        mu_base, _, _ = model(x, use_so=False)

                mu_fov, logvar, aux_fov = model(x, use_so=use_so_now)

                out = crit(x, mu_fov, logvar, gt, gtv_use, aux_fov, teacher_mask=mteach)
                loss = out["loss"]

                # Benefit map for self-supervision (target built without grads)
                loss_benefit = loss.new_zeros(())
                if do_benefit:
                    with torch.no_grad():
                        B = F.relu((mu_base - gt).abs() - (mu_fov.detach() - gt).abs())
                        if float(args.benefit_beta) > 0:
                            tilde_B = torch.clamp(B / float(args.benefit_beta), 0, 1)
                        else:
                            vv = (gtv_use > 0.5)
                            if vv.any():
                                q = torch.quantile(B[vv], float(args.benefit_q))
                                tilde_B = torch.clamp(B / (q + 1e-6), 0, 1)
                            else:
                                tilde_B = torch.clamp(B, 0, 1)

                        if int(args.benefit_blur_ks) > 1:
                            tilde_B = gaussian_blur2d(
                                tilde_B,
                                ks=int(args.benefit_blur_ks),
                                sigma=float(args.benefit_blur_sigma),
                            )

                    bce = F.binary_cross_entropy_with_logits(aux_fov["bfs_logits"], tilde_B, reduction="none")
                    denom = gtv_use.sum().clamp_min(1.0)
                    loss_benefit = (bce * gtv_use).sum() / denom

                total_raw = loss + args.w_benefit * loss_benefit
                total = total_raw / float(accum)

            scaler.scale(total).backward()

            # optimizer step on accumulation boundary
            if (it % accum == 0) or (it == len(dl_tr)):
                # grad clipping
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)

            # metrics
            m_valid = apply_eval_range(gt, gtv, args.eval_min, args.eval_max) if args.train_log_eval_range else gtv
            m = depth_metrics(mu_fov.detach(), gt, m_valid)

            loss_meter += float(total_raw.detach().item())
            mae_meter += m["mae_m"]
            nll_meter += out["nll"]
            md_meter += out["mask_distill"]
            benefit_meter += float(loss_benefit.detach().item())

            lrs = [float(pg["lr"]) for pg in opt.param_groups]
            lr_show = f"{min(lrs):.1e}~{max(lrs):.1e}" if len(lrs) > 1 else f"{lrs[0]:.1e}"
            pbar.set_postfix(
                {
                    "ep": ep,
                    "loss": f"{loss_meter/it:.3f}",
                    "mae(mm)": f"{(mae_meter/it)*1000:.1f}",
                    "nll": f"{nll_meter/it:.3f}",
                    "md": f"{md_meter/it:.3f}",
                    "benefit": f"{benefit_meter/it:.3f}",
                    "w_nll": f"{out['w_nll']:.2f}",
                    "lr": lr_show,
                }
            )

        # ---- epoch train summary ----
        train_dt = time.time() - t_ep0
        lrs = [float(pg["lr"]) for pg in opt.param_groups]
        lr_show = f"{min(lrs):.1e}~{max(lrs):.1e}" if len(lrs) > 1 else f"{lrs[0]:.1e}"
        print(
            f"   >>> Train: loss={loss_meter/it:.3f} mae(mm)={(mae_meter/it)*1000:.1f} "
            f"nll={nll_meter/it:.3f} md={md_meter/it:.3f} benefit={benefit_meter/it:.3f} "
            f"| lr={lr_show} | time={train_dt:.1f}s"
        )

        # ---- Val ----
        t0 = time.time()
        eval_model = ema.ema if (ema is not None and args.ema_eval) else model
        val = evaluate(
            eval_model,
            dl_va,
            device,
            args.eval_min,
            args.eval_max,
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            teacher_enable=args.teacher_enable,
            teacher_thr=args.teacher_thr,
        )
        dt = time.time() - t0
        msg = (
            f"   >>> Val: mae(mm)={val['mae_m']*1000:.1f} rmse(mm)={val['rmse_m']*1000:.1f} "
            f"absrel={val['absrel']:.4f}"
        )
        if "mae_roi_m" in val:
            msg += f" | mae_roi(mm)={val['mae_roi_m']*1000:.1f}"
        msg += f" | time={dt:.1f}s"
        print(msg)

        if val["mae_m"] < best:
            best = val["mae_m"]
            save_model = eval_model if (ema is not None and args.ema_eval) else model
            ckpt = {
                "model": save_model.state_dict(),
                "model_raw": model.state_dict(),
                "args": vars(args),
            }
            if ema is not None:
                ckpt["ema"] = ema.ema.state_dict()
            torch.save(ckpt, best_path)
            msg_best = f"   Saved best: {best_path} (val_mae_m={best:.4f} | {best*1000:.1f}mm)"
            if "mae_roi_m" in val:
                msg_best += f" | roi(mm)={val['mae_roi_m']*1000:.1f}"
            print(msg_best)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/void_rgrgd")

    # split source (important for VOID benchmark protocol)
    p.add_argument(
        "--void_split",
        type=str,
        default="official",
        choices=["official", "scan"],
        help="official: read official train_*.txt under the density dir; scan: scan data/ folders (legacy, not official protocol).",
    )
    p.add_argument(
        "--density",
        type=str,
        default=None,
        help="If --root points to VOID release root, set --density to void_1500/void_500/void_150.",
    )
    p.add_argument(
        "--official_debug_txt",
        type=int,
        default=0,
        help="Print the first N official txt lines after path resolving (debug for path issues).",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    # data
    # Use explicit (H,W) to keep aspect ratio (recommended for VOID)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=512)
    # legacy (square) option kept for backward compatibility
    p.add_argument("--img_size", type=int, default=None, help="(legacy) if set, use square resize img_size x img_size")
    p.add_argument("--depth_scale", type=float, default=256.0)
    p.add_argument("--no_sd_fill", action="store_true", help="disable nearest-neighbor filled depth input channel")
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--per_scene_cap", type=int, default=0)
    p.add_argument("--max_total", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_mode", type=str, default="scene", choices=["scene", "frame"])

    # augmentation
    p.add_argument("--aug_flip_p", type=float, default=0.5)
    p.add_argument("--aug_color_p", type=float, default=0.3)
    p.add_argument("--aug_rot_p", type=float, default=0.2)
    p.add_argument("--aug_rot_deg", type=float, default=10.0)

    # model
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--bp_iters", type=int, default=3)
    p.add_argument("--bp_beta", type=float, default=0.2, help="neighbor precision scale in BP (prevents overconfidence)")
    p.add_argument("--bp_refine_var", action="store_true", help="also refine logvar in BP (harder to calibrate)")
    p.add_argument("--no_residual", action="store_true", help="predict absolute depth instead of residual over filled depth")
    p.add_argument("--no_hard_sparse_copy", action="store_true", help="do not hard-copy sparse depth samples into output")
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--min_sigma", type=float, default=0.02)
    p.add_argument("--max_sigma", type=float, default=10.0)
    p.add_argument("--sofa_bias_scale", type=float, default=2.0)
    p.add_argument("--proxy_logit_scale", type=float, default=1.5)
    p.add_argument("--no_kmap", action="store_true")

    # ViT
    p.add_argument("--vit_name", type=str, default="vit_small_patch16_224")
    p.add_argument("--vit_patch", type=int, default=16)
    p.add_argument("--vit_no_pretrained", action="store_true")
    p.add_argument("--vit_local_weights", type=str, default="", help="Optional path to local ViT weights (.safetensors/.pth/.pt).")

    # YOLO (keep for optional online; but for direction-A inference you set none)
    p.add_argument("--yolo_mode", type=str, default="none", choices=["none", "online"])
    p.add_argument("--yolo_ckpt", type=str, default="yolov8n.pt")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.7)
    p.add_argument("--yolo_small_area_max", type=int, default=48 * 48)
    p.add_argument("--yolo_weight", type=float, default=0.4)

    # transformer backbone
    p.add_argument("--win_size", type=int, default=8)
    p.add_argument("--tf_depth", type=int, default=2)

    # uncertainty init
    p.add_argument("--init_logvar", type=float, default=-2.0)

    # BFS-Head params
    p.add_argument("--bfs_p", type=float, default=0.05)  # target mass proportion
    p.add_argument("--bfs_t", type=float, default=0.1)   # temperature

    # train
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    p.add_argument("--num_workers", type=int, default=4)

    # LR + warmup (requested)
    p.add_argument("--lr", type=float, default=3e-4)           # was 2e-4
    p.add_argument("--lr_warmup_epochs", type=int, default=5) # warmup 5~10
    p.add_argument("--lr_min_ratio", type=float, default=0.05, help="cosine LR min ratio after warmup (relative to --lr)")
    p.add_argument("--lr_vit", type=float, default=1e-5, help="LR for ViT backbone (usually smaller)")
    p.add_argument("--freeze_vit_epochs", type=int, default=0, help="freeze ViT for first N epochs")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)     # requested clip

    # loss weights
    p.add_argument("--w_smooth", type=float, default=0.03)
    p.add_argument("--w_so", type=float, default=1.0)
    p.add_argument("--w_mask_sparse", type=float, default=1e-4)  # slightly lower
    p.add_argument("--w_calib", type=float, default=0.0)
    p.add_argument("--so_gamma", type=float, default=2.0)
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_sparse", type=float, default=0.1)
    p.add_argument("--w_sigma_obs", type=float, default=0.02)
    # NLL warmup: start 0 -> max
    p.add_argument("--w_nll_max", type=float, default=0.05)        # stable default; set 0.5 if you want
    p.add_argument("--nll_warmup_epochs", type=int, default=20)

    p.add_argument("--sparse_agree_tau", type=float, default=0.2)

    # distillation (Direction-A)
    p.add_argument("--teacher_enable", action="store_true")
    p.add_argument("--teacher_subdir", type=str, default="yolo_mask")
    p.add_argument("--teacher_ext", type=str, default=".png")
    p.add_argument("--w_mask_distill", type=float, default=0.2)
    p.add_argument("--teacher_thr", type=float, default=0.5)

    # BFS loss weights
    p.add_argument("--w_mass", type=float, default=0.2)
    p.add_argument("--w_edgetv", type=float, default=0.02)
    p.add_argument("--w_scale", type=float, default=0.05)
    # detail preservation & mask stability
    p.add_argument("--w_grad", type=float, default=0.1, help="depth gradient matching loss weight (reduce over-smoothing)")
    p.add_argument("--grad_edge_k", type=float, default=2.0, help="edge weight multiplier for gradient loss")
    p.add_argument("--w_mask_coh", type=float, default=0.02, help="mask local coherence loss weight (reduce speckles)")
    p.add_argument("--coh_ks", type=int, default=7, help="kernel size for mask coherence loss")
    p.add_argument("--mask_blur_ks", type=int, default=7, help="gaussian blur ks for BFS logits (odd; 1 disables)")
    p.add_argument("--mask_blur_sigma", type=float, default=2.0, help="gaussian blur sigma for BFS logits")
    p.add_argument("--no_so_budget_norm", action="store_true", help="disable per-image budget normalization for mask_so")
    # benefit target shaping (benefit_beta<=0 uses quantile)
    p.add_argument("--benefit_q", type=float, default=0.95, help="quantile used when benefit_beta<=0")
    p.add_argument("--benefit_blur_ks", type=int, default=7, help="gaussian blur ks for benefit target (odd; 1 disables)")
    p.add_argument("--benefit_blur_sigma", type=float, default=2.0, help="gaussian blur sigma for benefit target")
    p.add_argument("--w_benefit", type=float, default=0.1)
    p.add_argument("--benefit_beta", type=float, default=0.15)  # normalization for benefit map
    p.add_argument("--benefit_every", type=int, default=4, help="compute benefit supervision every N iterations (1=every iter)")
    p.add_argument("--so_warmup_epochs", type=int, default=5, help="train without SOFA/BFS for first N epochs")

    # eval
    p.add_argument("--eval_min", type=float, default=0.2)
    p.add_argument("--eval_max", type=float, default=5.0)
    p.add_argument("--tta_steps", type=int, default=0)
    p.add_argument("--tta_lr", type=float, default=5e-2)

    # --- multi-frame fusion (cheap, no flow) ---
    p.add_argument(
        "--temporal_radius",
        type=int,
        default=0,
        help="fuse +-k neighboring frames into the *filled depth hint* (0=disable). "
             "Only affects hint channel; sd/vm remain current-frame.",
    )
    p.add_argument(
        "--temporal_mode",
        type=str,
        default="union",
        choices=["union", "avg"],
        help="union: fill missing pixels from neighbors (safest); avg: average (may ghost)",
    )

    # --- ROI crop (teacher-guided zoom-in) ---
    p.add_argument(
        "--roi_crop_p",
        type=float,
        default=0.0,
        help="probability of teacher-guided ROI crop (zoom-in). Works best with --teacher_enable.",
    )
    p.add_argument("--roi_crop_scale_min", type=float, default=0.55)
    p.add_argument("--roi_crop_scale_max", type=float, default=0.95)
    p.add_argument("--roi_crop_thr", type=float, default=0.5)

    # --- CSPN-lite refinement ---
    p.add_argument("--no_cspn", action="store_true", help="disable CSPN-lite post refinement")
    p.add_argument("--cspn_iters", type=int, default=4)
    p.add_argument("--cspn_hidden", type=int, default=24)

    # --- EMA ---
    p.add_argument("--ema_enable", action="store_true", help="enable EMA of model weights")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_eval", action="store_true", help="evaluate/save using EMA weights")

    # consistency toggles
    # Default to True for VOID so train/val are directly comparable under eval_min/eval_max.
    p.add_argument(
        "--train_apply_eval_range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply eval_min/max mask to training loss valid pixels (recommended for VOID)",
    )
    p.add_argument(
        "--train_log_eval_range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="log training MAE using eval_min/max mask for direct comparability",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
