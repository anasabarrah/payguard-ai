"""Explain a risk score: top-3 SHAP reasons, then one plain-English sentence.

SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`), which
is exact for tree models. The sentence is written by an LLM on Groq from those
reasons only; every number in it must appear in the reasons, otherwise (or with
no GROQ_API_KEY, or on any API error) a fill-in-the-blanks template is used.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd
import requests
from dotenv import load_dotenv

from payguard.features import FEATURES, SELLER_COUNTRY
from payguard.risk import RiskModel

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass
class Reason:
    feature: str
    value: object
    shap: float       # contribution to the log-odds of fraud
    text: str         # factual description with the exact numbers

    @property
    def raises_risk(self) -> bool:
        return self.shap > 0


@dataclass
class Explanation:
    score: float
    level: str
    reasons: list[Reason]
    sentence: str
    source: str       # "llm" or "template"


def describe(feature: str, value) -> str:
    if feature == "amount_ratio":
        return f"The amount is {value:.1f}x the median payment of the previous 24 hours"
    if feature == "payer_txn_24h":
        return f"The payer made {int(value)} other payments in the previous 24 hours"
    if feature == "payer_is_new":
        return "This is the first payment from this payer" if value else "The payer has paid before"
    if feature == "country":
        if value == SELLER_COUNTRY:
            return f"The payer is in the seller's country ({value})"
        if value == "OTHER":
            return "The payer is in a less common country"
        return f"The payer is in {value}, a cross-border payment"
    raise ValueError(feature)


def top_reasons(model: RiskModel, features: pd.DataFrame, k: int = 3) -> list[list[Reason]]:
    """Top-k reasons per row: the features that pushed the score up the most.

    If fewer than k features raise the risk, the remaining slots go to the
    features that lowered it the most, so a low score is explained too.
    Features with no effect on this score are never listed as reasons, so a
    row can have fewer than k reasons.
    """
    contrib = model.booster.predict(features[FEATURES], pred_contrib=True)[:, :-1]
    out = []
    for i, row in enumerate(contrib):
        order = sorted(range(len(FEATURES)), key=lambda j: row[j], reverse=True)
        up = [j for j in order if row[j] > 0][:k]
        down = sorted((j for j in order if row[j] < 0), key=lambda j: row[j])[: k - len(up)]
        out.append([Reason(FEATURES[j], _plain(features.iloc[i][FEATURES[j]]), float(row[j]),
                           describe(FEATURES[j], _plain(features.iloc[i][FEATURES[j]])))
                    for j in up + down])
    return out


# --- sentence -------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "twice": 2, "double": 2, "triple": 3, "half": 0.5, "dozen": 12,
}


def numbers_in(text: str) -> set[float]:
    """Every number in text, written with digits or as an English word."""
    nums = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}
    words = re.findall(r"[a-z]+", text.lower())
    nums |= {float(_NUMBER_WORDS[w]) for w in words if w in _NUMBER_WORDS}
    return nums


def allowed_numbers(score: float, reasons: list[Reason]) -> set[float]:
    facts = _facts(score, "", reasons)
    return numbers_in(facts) | {3.0}  # "3 reasons" is harmless


def is_grounded(sentence: str, score: float, reasons: list[Reason]) -> bool:
    return numbers_in(sentence) <= allowed_numbers(score, reasons)


def template_sentence(score: float, level: str, reasons: list[Reason]) -> str:
    up = [r.text[0].lower() + r.text[1:] for r in reasons if r.raises_risk]
    if not up:
        return f"Risk score {score:.1f}/100 ({level}): nothing in this payment stands out as risky."
    return f"Risk score {score:.1f}/100 ({level}) mainly because {_join(up)}."


def llm_sentence(score: float, level: str, reasons: list[Reason], timeout: float = 15.0) -> str | None:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    prompt = (
        "You explain fraud alerts to an online seller. Write ONE short sentence in plain "
        "English saying why this payment got its risk score. Use only the "
        "facts below. Copy every number exactly as written, using digits; do not round, "
        "compute or add any other number. No preamble, no quotes.\n\n"
        + _facts(score, level, reasons)
    )
    try:
        resp = requests.post(GROQ_URL, timeout=timeout,
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0.2, "reasoning_effort": "low",
                                   # gpt-oss reasons before answering; leave room for both.
                                   "max_completion_tokens": 2000})
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip().strip('"')
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None
    return text or None


def explain(score: float, level: str, reasons: list[Reason], use_llm: bool = True) -> Explanation:
    for _ in range(2 if use_llm else 0):
        text = llm_sentence(score, level, reasons)
        if text and is_grounded(text, score, reasons):
            return Explanation(score, level, reasons, text, "llm")
    return Explanation(score, level, reasons, template_sentence(score, level, reasons), "template")


def _facts(score: float, level: str, reasons: list[Reason]) -> str:
    lines = [f"Risk score: {score:.1f} out of 100" + (f" ({level} risk)" if level else "")]
    for r in reasons:
        lines.append(f"- {r.text} ({'raises' if r.raises_risk else 'lowers'} the risk)")
    return "\n".join(lines)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _plain(v):
    return v.item() if hasattr(v, "item") else v
