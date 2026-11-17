"""Train and evaluate the Random Forest (write-up 3.16.1 to 3.16.4, 3.17).

Usage:
    python train_model.py                 # uses ./data, downloads the dataset if missing
    python train_model.py --data path     # folder of .parquet files OR class sub-folders

Dataset: "Alzheimer MRI" (6,400 axial slices, 128x128), public copy at
huggingface.co/datasets/Falah/Alzheimer_MRI. Its four labels are mapped to the
two classes of this study:
    No AD = Non_Demented
    AD    = Very_Mild_Demented + Mild_Demented + Moderate_Demented

Outputs: models/rf_model.joblib and models/metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from mindtrace.ml import features, preprocess

ROOT = Path(__file__).parent
SEED = 42
STAGES = {0: "Mild", 1: "Moderate", 2: "Non-demented", 3: "Very mild"}
HF = "https://huggingface.co/datasets/Falah/Alzheimer_MRI/resolve/main/data/"
HF_FILES = ["train-00000-of-00001-c08a401c53fe5312.parquet", "test-00000-of-00001-44110b9df98c5585.parquet"]
FOLDER_STAGE = {"milddemented": 0, "moderatedemented": 1, "nondemented": 2, "verymilddemented": 3}


def load_images(data_dir: Path) -> tuple[list[bytes], np.ndarray]:
    """Returns (encoded image bytes, 4-class stage label)."""
    parquets = sorted(data_dir.rglob("*.parquet"))
    if parquets:
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        return [d["bytes"] for d in df["image"]], df["label"].to_numpy()
    blobs, labels = [], []
    for p in sorted(data_dir.rglob("*")):
        key = p.parent.name.lower().replace("_", "").replace(" ", "")
        if p.is_file() and preprocess.allowed_file(p.name) and key in FOLDER_STAGE:
            blobs.append(p.read_bytes())
            labels.append(FOLDER_STAGE[key])
    if blobs:
        return blobs, np.array(labels)
    print("No dataset found, downloading from Hugging Face ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in HF_FILES:
        urllib.request.urlretrieve(HF + f, data_dir / f)
    return load_images(data_dir)


def metrics_at(y, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(y, proba),
        "confusion": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # 1. Collection and screening ---------------------------------------------
    blobs, stage = load_images(Path(args.data))
    seen, keep, rejected = set(), [], 0
    for i, b in enumerate(blobs):
        h = hashlib.md5(b).hexdigest()
        ok = h not in seen and cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) is not None
        seen.add(h)
        keep.append(ok)
        rejected += not ok
    keep = np.array(keep)
    blobs, stage = [b for b, k in zip(blobs, keep) if k], stage[keep]
    y = (stage != 2).astype(int)
    print(f"{len(blobs)} usable images ({rejected} duplicate/corrupt removed): AD={y.sum()}  No AD={(1 - y).sum()}")

    # 2. Preprocessing + feature extraction -----------------------------------
    t0 = time.time()
    X = np.array([[v for v in features.extract(preprocess.preprocess(
        cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE))).values()] for b in blobs])
    names = features.FEATURE_NAMES
    print(f"Feature matrix {X.shape} in {time.time() - t0:.0f}s")

    # 3. Split 70 / 15 / 15, stratified on disease stage ----------------------
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=stage, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=stage[rest], random_state=SEED)

    # 4. Model selection on the validation set --------------------------------
    grid = [dict(n_estimators=n, max_depth=d, min_samples_leaf=l, max_features=f)
            for n in (200, 400) for d in (None, 16) for l in (1, 2, 4) for f in ("sqrt", 0.33)]
    trials, best = [], None
    for g in grid:
        m = RandomForestClassifier(**g, n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
        acc = accuracy_score(y[va], m.predict(X[va]))
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        trials.append({**g, "val_accuracy": acc, "val_auc": auc})
        if best is None or (acc, auc) > (best[1], best[2]):
            best = (g, acc, auc, m)
    params, val_acc, val_auc, model = best
    train_acc = accuracy_score(y[tr], model.predict(X[tr]))
    print(f"Best: {params}  train={train_acc:.3f} val={val_acc:.3f}")

    cv = cross_val_score(RandomForestClassifier(**params, n_jobs=-1, random_state=SEED), X[tr], y[tr],
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")

    # 5. Final evaluation on the reserved test set ----------------------------
    proba = model.predict_proba(X[te])[:, 1]
    test = metrics_at(y[te], proba)
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y[te], proba)
    step = max(1, len(fpr) // 60)
    by_stage = [{"stage": STAGES[s], "n": int((stage[te] == s).sum()),
                 "correct": int((pred[stage[te] == s] == y[te][stage[te] == s]).sum())} for s in (2, 3, 0, 1)]

    perm = permutation_importance(model, X[te], y[te], n_repeats=5, random_state=SEED, n_jobs=-1)
    order = np.argsort(model.feature_importances_)[::-1]
    importance = [{"name": names[i], **features.describe(names[i]),
                   "importance": float(model.feature_importances_[i]),
                   "permutation": float(perm.importances_mean[i])} for i in order]

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "python_version": sys.version.split()[0],
        "dataset": {"name": "Alzheimer MRI (Falah/Alzheimer_MRI, 4-class, mapped to binary)",
                    "total": int(len(y)), "ad": int(y.sum()), "no_ad": int((1 - y).sum()), "removed": int(rejected),
                    "stages": {STAGES[s]: int((stage == s).sum()) for s in (2, 3, 0, 1)},
                    "split": {"train": len(tr), "validation": len(va), "test": len(te)}},
        "pipeline": {"image_size": preprocess.IMG_SIZE, "gray_levels": preprocess.GRAY_LEVELS,
                     "distances": features.DISTANCES, "angles": [0, 45, 90, 135], "n_features": len(names)},
        "params": params, "trials": trials,
        "train_accuracy": train_acc, "validation": {"accuracy": val_acc, "roc_auc": val_auc},
        "cv": {"folds": 5, "mean": float(cv.mean()), "std": float(cv.std()), "scores": cv.tolist()},
        "test": test, "by_stage": by_stage,
        "roc": {"fpr": fpr[::step].tolist() + [1.0], "tpr": tpr[::step].tolist() + [1.0]},
        "feature_names": names, "feature_importance": importance,
        "group_importance": {g: float(sum(f["importance"] for f in importance if f["group"] == g))
                             for g in ("GLCM texture", "Intensity")},
        "class_means": {"ad": dict(zip(names, X[tr][y[tr] == 1].mean(0).tolist())),
                        "no_ad": dict(zip(names, X[tr][y[tr] == 0].mean(0).tolist()))},
        "feature_range": dict(zip(names, np.percentile(X[tr], [1, 99], axis=0).T.tolist())),
    }
    joblib.dump(model, out / "rf_model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.save(out / "test_indices.npy", te)
    c = test["confusion"]
    print(f"\nTEST  accuracy={test['accuracy']:.4f} precision={test['precision']:.4f} recall={test['recall']:.4f} "
          f"f1={test['f1']:.4f} auc={test['roc_auc']:.4f}\n      TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
    print(f"5-fold CV accuracy {cv.mean():.4f} +/- {cv.std():.4f}")
    print(f"Saved {out / 'rf_model.joblib'} ({(out / 'rf_model.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
