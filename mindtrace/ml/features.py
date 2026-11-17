"""Handcrafted feature extraction: GLCM texture + intensity statistics.

GLCM follows Haralick, Shanmugam & Dinstein (1973). The matrix is built with
NumPy only, so the pipeline needs nothing beyond OpenCV and NumPy.

Feature vector (69 values):
  21 texture   7 Haralick descriptors x 3 pixel distances (1, 2, 4), each the
               mean over four directions (0, 45, 90, 135 degrees)
  21 texture   the spread (max - min) of each descriptor across the four
               directions, which captures how directional the texture is
  11 intensity summary statistics of the tissue pixels
  16 intensity a 16-bin histogram of the tissue pixels
"""
from __future__ import annotations

import numpy as np

from .preprocess import GRAY_LEVELS, quantise

DISTANCES = (1, 2, 4)
# (row step, column step) for 0, 45, 90 and 135 degrees
ANGLES = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")
TISSUE_THRESHOLD = 20   # gray value above which a pixel counts as tissue
HIST_BINS = 16

_I, _J = np.meshgrid(np.arange(GRAY_LEVELS), np.arange(GRAY_LEVELS), indexing="ij")
_DIFF = (_I - _J).astype(np.float64)


def glcm(q: np.ndarray, dr: int, dc: int, levels: int = GRAY_LEVELS) -> np.ndarray:
    """Symmetric, normalised gray-level co-occurrence matrix for one offset."""
    h, w = q.shape
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    a = q[r0:r1, c0:c1].ravel()
    b = q[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel()
    m = np.bincount(a * levels + b, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    m += m.T
    return m / m.sum()


def glcm_props(p: np.ndarray) -> dict[str, float]:
    """Haralick descriptors of one normalised co-occurrence matrix."""
    mu_i, mu_j = (p * _I).sum(), (p * _J).sum()
    sd_i = np.sqrt((p * (_I - mu_i) ** 2).sum())
    sd_j = np.sqrt((p * (_J - mu_j) ** 2).sum())
    asm = (p ** 2).sum()
    nz = p[p > 0]
    return {
        "contrast": (p * _DIFF ** 2).sum(),
        "dissimilarity": (p * np.abs(_DIFF)).sum(),
        "homogeneity": (p / (1.0 + _DIFF ** 2)).sum(),
        "energy": np.sqrt(asm),
        "correlation": ((p * (_I - mu_i) * (_J - mu_j)).sum() / (sd_i * sd_j)) if sd_i * sd_j > 1e-12 else 1.0,
        "asm": asm,
        "entropy": float(-(nz * np.log2(nz)).sum()),
    }


def texture_features(gray: np.ndarray) -> dict[str, float]:
    q = quantise(gray)
    out: dict[str, float] = {}
    for d in DISTANCES:
        per_angle = [glcm_props(glcm(q, dr * d, dc * d)) for dr, dc in ANGLES]
        for name in PROPS + ("entropy",):
            vals = [p[name] for p in per_angle]
            out[f"glcm_{name}_d{d}"] = float(np.mean(vals))
            out[f"glcmdir_{name}_d{d}"] = float(np.max(vals) - np.min(vals))
    return out


def intensity_features(gray: np.ndarray) -> dict[str, float]:
    g = gray.astype(np.float64)
    tissue = g[g > TISSUE_THRESHOLD]
    if tissue.size < 16:
        tissue = g.ravel()
    mean, std = tissue.mean(), tissue.std()
    z = (tissue - mean) / std if std > 1e-9 else np.zeros_like(tissue)
    p10, p25, p50, p75, p90 = np.percentile(tissue, [10, 25, 50, 75, 90])
    hist = np.bincount(tissue.astype(np.intp), minlength=256) / tissue.size
    nz = hist[hist > 0]
    bins = np.histogram(tissue, bins=HIST_BINS, range=(TISSUE_THRESHOLD, 256))[0] / tissue.size
    stats = {
        "int_mean": mean,
        "int_std": std,
        "int_skewness": float((z ** 3).mean()),
        "int_kurtosis": float((z ** 4).mean() - 3.0),
        "int_median": p50,
        "int_p10": p10,
        "int_p90": p90,
        "int_iqr": p75 - p25,
        "int_entropy": float(-(nz * np.log2(nz)).sum()),
        "int_tissue_fraction": tissue.size / g.size,
        "int_bright_fraction": float((g > 170).mean()),
    }
    stats.update({f"hist_bin{i + 1:02d}": float(v) for i, v in enumerate(bins)})
    return stats


def extract(gray: np.ndarray) -> dict[str, float]:
    """Preprocessed 128x128 uint8 image -> ordered feature dictionary."""
    feats = texture_features(gray)
    feats.update(intensity_features(gray))
    return {k: float(v) for k, v in feats.items()}


FEATURE_NAMES = list(extract(np.random.default_rng(0).integers(0, 255, (128, 128)).astype(np.uint8)).keys())

_PROP_LABEL = {
    "contrast": "Contrast", "dissimilarity": "Dissimilarity", "homogeneity": "Homogeneity",
    "energy": "Energy", "correlation": "Correlation", "asm": "Angular second moment",
    "entropy": "Entropy",
}
_INT_LABEL = {
    "int_mean": "Mean intensity", "int_std": "Intensity spread (std)", "int_skewness": "Skewness",
    "int_kurtosis": "Kurtosis", "int_median": "Median intensity", "int_p10": "10th percentile",
    "int_p90": "90th percentile", "int_iqr": "Interquartile range", "int_entropy": "Histogram entropy",
    "int_tissue_fraction": "Tissue area fraction", "int_bright_fraction": "Bright-pixel fraction",
}


def describe(name: str) -> dict[str, str]:
    """Human-readable label and group for a feature key."""
    if name.startswith("glcm_"):
        prop, d = name[5:].rsplit("_d", 1)
        return {"label": f"{_PROP_LABEL[prop]} · {d} px", "group": "GLCM texture"}
    if name.startswith("glcmdir_"):
        prop, d = name[8:].rsplit("_d", 1)
        return {"label": f"{_PROP_LABEL[prop]} directional spread · {d} px", "group": "GLCM texture"}
    if name.startswith("hist_bin"):
        i = int(name[8:])
        width = (256 - TISSUE_THRESHOLD) / HIST_BINS
        lo, hi = TISSUE_THRESHOLD + (i - 1) * width, TISSUE_THRESHOLD + i * width
        return {"label": f"Share of pixels in gray {lo:.0f} to {hi:.0f}", "group": "Intensity"}
    return {"label": _INT_LABEL[name], "group": "Intensity"}
