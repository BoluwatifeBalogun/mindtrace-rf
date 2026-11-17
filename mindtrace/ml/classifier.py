"""Prediction service: wraps the trained Random Forest (write-up 3.12.3, 3.12.4)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import features, preprocess

CLASSES = {0: "No Alzheimer's Disease", 1: "Alzheimer's Disease"}
SHORT = {0: "No AD", 1: "AD"}


class Classifier:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "rf_model.joblib")
        self.metrics = json.loads((model_dir / "metrics.json").read_text())
        self.feature_names = self.metrics["feature_names"]
        if self.feature_names != features.FEATURE_NAMES:
            raise RuntimeError("Model was trained with a different feature set. Run train_model.py again.")
        self.model.n_jobs = 1
        self.importance = {f["name"]: f["importance"] for f in self.metrics["feature_importance"]}

    def analyse(self, data: bytes) -> dict:
        """Raw upload bytes -> prediction record. Raises preprocess.InvalidImage."""
        t0 = time.perf_counter()
        img = preprocess.decode(data)
        preprocess.validate(img)
        gray = preprocess.preprocess(img)
        t1 = time.perf_counter()
        feats = features.extract(gray)
        t2 = time.perf_counter()
        x = np.array([[feats[n] for n in self.feature_names]])
        proba = self.model.predict_proba(x)[0]
        label = int(np.argmax(proba))
        t3 = time.perf_counter()
        return {
            "label": label,
            "class_name": CLASSES[label],
            "short": SHORT[label],
            "probability": float(proba[label]),
            "prob_ad": float(proba[1]),
            "prob_no_ad": float(proba[0]),
            "features": feats,
            "gray": gray,
            "timing_ms": {
                "preprocess": round((t1 - t0) * 1000, 1),
                "features": round((t2 - t1) * 1000, 1),
                "predict": round((t3 - t2) * 1000, 1),
                "total": round((t3 - t0) * 1000, 1),
            },
        }

    def top_features(self, feats: dict, k: int = 8) -> list[dict]:
        """The k most important model features with this scan's value beside the
        training-set average of each class. A model-level summary, not a heatmap."""
        means = self.metrics["class_means"]
        rows = []
        for f in self.metrics["feature_importance"][:k]:
            n = f["name"]
            lo, hi = self.metrics["feature_range"][n]
            span = (hi - lo) or 1.0
            pos = lambda v: round(min(max((v - lo) / span, 0), 1) * 100, 1)
            rows.append({
                "name": n, "label": f["label"], "group": f["group"],
                "importance": f["importance"],
                "value": feats[n],
                "mean_ad": means["ad"][n], "mean_no_ad": means["no_ad"][n],
                "pos": pos(feats[n]), "pos_ad": pos(means["ad"][n]), "pos_no_ad": pos(means["no_ad"][n]),
            })
        return rows
