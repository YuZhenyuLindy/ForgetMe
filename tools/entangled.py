# tools/entangled.py
# -*- coding: utf-8 -*-
"""
Entangled metric implementation.

- Entangled-D (paired): needs original image + background image + mask
- Entangled-S (single/unpaired): needs only background image + mask (alpha=0, beta=1)

All images are read as RGB in [0,1]. Mask is binary {0,1}, where 1 indicates the "inner" region (to-be-forgotten area).
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

EPS = 1e-6

def _load_img_as_float(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3 in [0,1]
    return arr

def _load_mask_as_bool(path: Path, thresh: float = 0.5) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.asarray(m).astype(np.float32)/255.0
    return (arr >= thresh)

def _region_flatten(x: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    # x: H,W,3 ; region_mask: H,W (bool)
    return x[region_mask].reshape(-1, 3)  # N,3

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: N,3 in [0,1]
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mean_var(x: np.ndarray) -> Tuple[float, float]:
    # x: N,3 ; merge channels as one population
    if x.size == 0:
        return 0.0, 0.0
    x_flat = x.reshape(-1)
    return float(np.mean(x_flat)), float(np.var(x_flat))

def _harmonic(a: float, b: float) -> float:
    denom = a + b + EPS
    return float(2.0 * a * b / denom)

def _combine_similarity(S_in: float, S_out: float) -> float:
    # Paper uses:  S_inner,outer = 2 * S_inner * (1 - S_outer) / (S_inner + (1 - S_outer) + eps)
    return float(2.0 * S_in * (1.0 - S_out) / (S_in + (1.0 - S_out) + EPS))

def _consistency(inner: np.ndarray, outer: np.ndarray) -> float:
    mu_in, var_in = _mean_var(inner)
    mu_out, var_out = _mean_var(outer)
    M = (2.0 * mu_in * mu_out) / (mu_in**2 + mu_out**2 + EPS)
    V = (2.0 * np.sqrt(max(var_in, 0.0)) * np.sqrt(max(var_out, 0.0))) / (var_in + var_out + EPS)
    # Note: Paper defines V using sigma; we use sqrt(var) here for numerical clarity.
    return float(M * V)

def entangled_paired(
    img_orig: np.ndarray,
    img_bg: np.ndarray,
    mask_inner: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Paired version (Entangled-D).

    img_orig, img_bg: H,W,3 in [0,1]
    mask_inner: H,W bool  (True = inner region)
    """
    assert img_orig.shape == img_bg.shape, "orig and bg size mismatch"
    assert img_orig.shape[:2] == mask_inner.shape, "mask size mismatch"
    mask_outer = ~mask_inner

    # Split regions
    X_in  = _region_flatten(img_orig, mask_inner)
    Y_in  = _region_flatten(img_bg,   mask_inner)
    X_out = _region_flatten(img_orig, mask_outer)
    Y_out = _region_flatten(img_bg,   mask_outer)

    # Similarity terms (use RMSE per paper; then convert outer to (1 - S_out))
    S_in  = 1.0 - _rmse(X_in, Y_in)   # higher is better (1 means identical)
    S_out = _rmse(X_out, Y_out)       # will be inverted inside combiner

    S = _combine_similarity(S_in, S_out)
    C = _consistency(Y_in, Y_out)     # consistency is computed on bg (retained) regions

    # Entangled = ((α+β) * S * C) / (α*C + β*S)
    num = (alpha + beta) * S * C
    den = (alpha * C + beta * S + EPS)
    return float(num / den)

def entangled_single(
    img_bg: np.ndarray,
    mask_inner: np.ndarray,
) -> float:
    """
    Single-image version (Entangled-S).
    Per paper: set alpha=0, beta=1 -> metric degenerates to consistency on bg regions.
    We report the same C(inner, outer) computed on background image.
    """
    assert img_bg.shape[:2] == mask_inner.shape, "mask size mismatch"
    mask_outer = ~mask_inner

    Y_in  = _region_flatten(img_bg, mask_inner)
    Y_out = _region_flatten(img_bg, mask_outer)

    C = _consistency(Y_in, Y_out)
    return float(C)

def entangled_from_files(
    path_bg: Path,
    path_mask: Path,
    path_orig: Optional[Path] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    img_bg = _load_img_as_float(path_bg)
    mask   = _load_mask_as_bool(path_mask)

    if path_orig is None:
        return entangled_single(img_bg, mask)
    else:
        img_orig = _load_img_as_float(path_orig)
        return entangled_paired(img_orig, img_bg, mask, alpha=alpha, beta=beta)
