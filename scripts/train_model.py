"""Train the PayGuard LightGBM risk model on PaySim with a temporal split.

    train: step <= TRAIN_END   validation: (TRAIN_END, VALID_END]   test: > VALID_END

Validation is used for early stopping and for choosing the action thresholds;
the test window is only used for the reported metrics. Also trains the same
model without the simulated `country` feature, so its contribution is visible.

Usage: python scripts/train_model.py [--data data/raw/paysim.csv]
Writes models/payguard_lgbm.txt and models/model_meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from payguard.features import FEATURES, build_paysim_features  # noqa: E402
from payguard.risk import to_score  # noqa: E402
from scripts.download_paysim import SHA256  # noqa: E402

TRAIN_END, VALID_END = 400, 550
SEED = 42
MONITOR_SHARE = 0.05  # review budget: flag ~5% of payments for monitoring
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    # Large leaves: PaySim payers rarely repeat (460 training rows with a prior
    # payment, 2 of them fraud), so smaller leaves learn noise from a handful of cases.
    "min_child_samples": 1000,
    "cat_smooth": 20,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "num_threads": 8,
}


def load_features(csv: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    raw = pd.read_csv(csv, usecols=["step", "type", "amount", "nameOrig", "isFraud"],
                      engine="pyarrow")
    df = build_paysim_features(raw, seed=SEED)
    df.to_parquet(cache)
    return df


# Risk may only grow with a larger ticket, more payments in 24h, or a new payer.
MONOTONE = {"amount_ratio": 1, "payer_txn_24h": 1, "payer_is_new": 1}


def fit(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str]) -> lgb.Booster:
    cats = [f for f in feats if f == "country"]
    params = {**PARAMS, "metric": "average_precision",
              "monotone_constraints": [MONOTONE.get(f, 0) for f in feats]}
    dtrain = lgb.Dataset(train[feats], train["isFraud"], categorical_feature=cats)
    dvalid = lgb.Dataset(valid[feats], valid["isFraud"], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=2000,
                     valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": round(float(threshold), 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def score_reference(p_valid: np.ndarray) -> dict:
    """Quantile grid of validation probabilities, finer in the top 1% where alerts live."""
    levels = np.unique(np.concatenate([np.linspace(0, 0.99, 991), np.linspace(0.99, 1, 1001)]))
    return {"levels": levels.round(6).tolist(),
            "probs": np.quantile(p_valid, levels).tolist()}


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """refund = max-F1 threshold; monitor = top MONITOR_SHARE of validation payments."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    refund = float(thr[np.argmax(f1)])
    monitor = float(np.quantile(p, 1 - MONITOR_SHARE))
    return min(monitor, refund), refund


def evaluate(model: lgb.Booster, feats: list[str], valid, test) -> dict:
    pv = model.predict(valid[feats])
    pt = model.predict(test[feats])
    yt = test["isFraud"].to_numpy()
    monitor, refund = choose_thresholds(valid["isFraud"].to_numpy(), pv)
    ref = score_reference(pv)
    return {
        "score_reference": ref,
        "best_iteration": model.best_iteration,
        "test_pr_auc": round(float(average_precision_score(yt, pt)), 4),
        "test_prevalence": round(float(yt.mean()), 5),
        "test_at_refund_threshold": metrics_at(yt, pt, refund),
        "test_at_monitor_threshold": metrics_at(yt, pt, monitor),
        "thresholds": {"monitor": monitor, "refund": refund},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "paysim.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    if not args.data.exists():
        print(f"{args.data} not found; run scripts/download_paysim.py first", file=sys.stderr)
        return 1

    df = load_features(args.data, ROOT / "data" / "paysim_features.parquet")
    train = df[df.step <= TRAIN_END]
    valid = df[(df.step > TRAIN_END) & (df.step <= VALID_END)]
    test = df[df.step > VALID_END]
    split = {name: {"rows": len(d), "frauds": int(d.isFraud.sum()),
                    "steps": [int(d.step.min()), int(d.step.max())]}
             for name, d in [("train", train), ("valid", valid), ("test", test)]}
    print(json.dumps(split, indent=2))

    model = fit(train, valid, FEATURES)
    full = evaluate(model, FEATURES, valid, test)
    no_country_feats = [f for f in FEATURES if f != "country"]
    ablation = evaluate(fit(train, valid, no_country_feats), no_country_feats, valid, test)

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "payguard_lgbm.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    meta_ref = full["score_reference"]
    gain = model.feature_importance("gain", iteration=model.best_iteration)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "PaySim (ealaxi/paysim1)", "sha256": SHA256,
                    "types": ["TRANSFER", "CASH_OUT"]},
        "features": FEATURES,
        "split": split,
        "params": {**PARAMS, "monotone_constraints": MONOTONE},
        # Score = percentile (0-100) of the model output among validation payments.
        "score_reference": full.pop("score_reference"),
        "score_thresholds": {k: round(float(to_score(v, meta_ref)), 2)
                             for k, v in full["thresholds"].items()},
        "metrics": full,
        "metrics_without_country": {k: v for k, v in ablation.items() if k != "score_reference"},
        "feature_importance_gain": {f: round(float(g), 1) for f, g in zip(FEATURES, gain)},
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"score_thresholds": meta["score_thresholds"], "with_country": full,
                      "without_country": meta["metrics_without_country"]}, indent=2))
    print(f"saved {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
