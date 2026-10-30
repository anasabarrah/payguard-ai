"""Load the trained LightGBM model and turn payments into 0-100 risk scores.

The score is the percentile of the model's fraud probability among the
validation payments: a score of 97 means "riskier than 97% of reference
payments". This stays readable even though PaySim's fraud rate drifts a lot
over time, which makes raw probabilities poorly calibrated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from payguard.features import FEATURES, features_from_payments

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def to_score(prob, reference: dict) -> np.ndarray:
    """Map probabilities to 0-100 percentiles using the stored quantile grid."""
    probs = np.asarray(reference["probs"])
    levels = np.asarray(reference["levels"])
    # Tree models output many tied values; keep the highest level per probability.
    xp, idx = np.unique(probs[::-1], return_index=True)
    fp = levels[::-1][idx]
    return np.round(100 * np.interp(np.asarray(prob, dtype=float), xp, fp), 1)


@dataclass
class RiskModel:
    booster: lgb.Booster
    meta: dict

    @classmethod
    def load(cls, model_dir: str | Path = MODEL_DIR) -> "RiskModel":
        model_dir = Path(model_dir)
        booster = lgb.Booster(model_file=str(model_dir / "payguard_lgbm.txt"))
        meta = json.loads((model_dir / "model_meta.json").read_text())
        return cls(booster, meta)

    @property
    def thresholds(self) -> dict:
        return self.meta["score_thresholds"]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(features[FEATURES])

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return to_score(self.predict_proba(features), self.meta["score_reference"])

    def level(self, score: float) -> str:
        if score >= self.thresholds["refund"]:
            return "high"
        if score >= self.thresholds["monitor"]:
            return "medium"
        return "low"

    def score_payments(self, payments: list[dict] | pd.DataFrame) -> pd.DataFrame:
        """Features + score + level for each payment (e.g. Ledger.list_payments())."""
        feats = features_from_payments(payments)
        out = feats.copy()
        out["risk_score"] = self.score(feats) if len(feats) else []
        out["risk_level"] = [self.level(s) for s in out["risk_score"]]
        return out
