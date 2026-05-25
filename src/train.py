import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .features import combine_pair
from .fingerprints import smiles_to_fingerprint

REQUIRED_COLS = {"smiles_a", "smiles_b", "interacts"}


def featurize(df: pd.DataFrame, cache: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    cache = {} if cache is None else cache
    feats, labels = [], []
    for _, row in df.iterrows():
        for key in ("smiles_a", "smiles_b"):
            if row[key] not in cache:
                cache[row[key]] = smiles_to_fingerprint(row[key])
        feats.append(combine_pair(cache[row["smiles_a"]], cache[row["smiles_b"]]))
        labels.append(int(row["interacts"]))
    return np.vstack(feats), np.array(labels, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    print(f"featurizing {len(df)} pairs (unique SMILES: {pd.concat([df.smiles_a, df.smiles_b]).nunique()})")
    X, y = featurize(df)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    aucs, aps, f1s = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        scores = clf.predict_proba(X[va])[:, 1]
        preds = (scores >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[va], scores))
        aps.append(average_precision_score(y[va], scores))
        f1s.append(f1_score(y[va], preds))
        print(f"fold {fold}: AUC={aucs[-1]:.3f} AP={aps[-1]:.3f} F1={f1s[-1]:.3f}")

    # Refit on full data
    final = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.7,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    final.fit(X, y)
    train_preds = (final.predict_proba(X)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y, train_preds).tolist()

    joblib.dump({"model": final}, out / "best_model.joblib")
    (out / "cv_results.json").write_text(json.dumps({
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)),
        "f1_mean": float(np.mean(f1s)),
        "train_confusion": cm,
        "n_pairs": int(len(df)),
    }, indent=2))


if __name__ == "__main__":
    main()
