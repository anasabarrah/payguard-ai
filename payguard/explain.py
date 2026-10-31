"""Explain a risk score: top-3 SHAP reasons, then one plain-English sentence.

SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`), which
is exact for tree models. The sentence is written by an LLM on Groq from those
reasons only; every number in it must appear in the reasons, and it must not
name a country (only "cross-border" or "domestic"), otherwise (or with no
GROQ_API_KEY, or on any API error) a fill-in-the-blanks template is used.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd
import requests
from dotenv import load_dotenv

from payguard.features import FEATURES
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


def describe(feature: str, value, country: str = "") -> str:
    if feature == "amount_ratio":
        return f"The amount is {value:.1f}x the median payment of the previous 24 hours"
    if feature == "payer_txn_24h":
        return f"The payer made {int(value)} other payments in the previous 24 hours"
    if feature == "payer_is_new":
        return "This is the first payment from this payer" if value else "The payer has paid before"
    if feature == "cross_border":
        where = f" (payer in {country})" if country else ""
        if value:
            return f"This is a cross-border payment{where}"
        return f"This is a domestic payment{where}"
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
        country = features["payer_country"].iloc[i] if "payer_country" in features else ""
        out.append([Reason(FEATURES[j], _plain(features.iloc[i][FEATURES[j]]), float(row[j]),
                           describe(FEATURES[j], _plain(features.iloc[i][FEATURES[j]]), country))
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


# ISO 3166-1 alpha-2 codes, matched only in capitals ("IN", not "in").
_COUNTRY_CODES = set("""
AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS
BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE
EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM
HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC
LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA
NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW
SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO
TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW UK
""".split()) - {"AI"}   # "AI" is far more likely to mean the technology
_COUNTRY_WORDS = re.compile(r"\b(" + "|".join("""
afghan\w* albania\w* algeria\w* argentin\w* australia\w* austria\w* bangladesh\w* belgi\w*
brazil\w* britain british canad\w* chile\w* china chinese colombia\w* egypt\w* england english
france french german\w* ghana\w* greece greek india\w* indonesia\w* iran\w* iraq\w* ireland irish
israel\w* ital\w* japan\w* kenya\w* korea\w* mexic\w* moroc\w* netherlands dutch nigeria\w*
norw\w* pakistan\w* philippin\w* filipino poland polish portug\w* romania\w* russia\w* saudi
scotland scottish singapor\w* spain spanish swed\w* swiss switzerland thai\w* turk\w* ukrain\w*
vietnam\w* america\w* usa emirates europe\w* african? asian? latin
""".split()) + r")\b", re.I)
_PAYER_LOCATION = re.compile(r"\b(payer|buyer|customer|card)s?\s+(?:is\s+|was\s+|located\s+)?"
                             r"(?:in|from|based\s+in)\b", re.I)


def names_country(text: str) -> bool:
    """True if a text names a country, a nationality or where the payer is.

    PayGuard's fairness rule: a sentence may say "cross-border" or "domestic",
    never which country (the country stays visible in the table).
    """
    return (any(tok in _COUNTRY_CODES for tok in re.findall(r"\b[A-Z]{2}\b", text))
            or bool(_COUNTRY_WORDS.search(text)) or bool(_PAYER_LOCATION.search(text)))


def neutral_text(r: "Reason") -> str:
    """The reason as the LLM and the template see it: without the payer's country."""
    return describe(r.feature, r.value)


def levels_in(text: str) -> set[str]:
    """Risk levels a text names, e.g. "high-risk" or "a medium risk score"."""
    return set(re.findall(r"\b(low|medium|high)(?:[\s-]+risk|\s+score)", text.lower()))


def is_grounded(sentence: str, score: float, reasons: list[Reason], level: str = "") -> bool:
    """Every number comes from the facts, no other risk level than the real one is named,
    and no country is named."""
    if level and not levels_in(sentence) <= {level}:
        return False
    if names_country(sentence):
        return False
    return numbers_in(sentence) <= allowed_numbers(score, reasons)


def template_sentence(score: float, level: str, reasons: list[Reason]) -> str:
    up = [t[0].lower() + t[1:] for t in (neutral_text(r) for r in reasons if r.raises_risk)]
    if not up:
        return f"Risk score {score:.1f}/100 ({level}): nothing in this payment stands out as risky."
    return f"Risk score {score:.1f}/100 ({level}) mainly because {_join(up)}."


def llm_sentence(score: float, level: str, reasons: list[Reason], timeout: float = 15.0) -> str | None:
    prompt = (
        "You explain fraud alerts to an online seller. Write ONE short sentence in plain "
        "English saying why this payment got its risk score. Use only the "
        "facts below. Copy every number exactly as written, using digits; do not round, "
        "compute or add any other number. Never mention countries or nationalities: say "
        "\"cross-border\" or \"domestic\". No preamble, no quotes.\n\n"
        + _facts(score, level, reasons)
    )
    return llm_complete(prompt, timeout=timeout)


def llm_complete(prompt: str, timeout: float = 15.0, max_tokens: int = 2000) -> str | None:
    """One Groq chat completion; None without GROQ_API_KEY or on any API error."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        resp = requests.post(GROQ_URL, timeout=timeout,
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0.2, "reasoning_effort": "low",
                                   # gpt-oss reasons before answering; leave room for both.
                                   "max_completion_tokens": max_tokens})
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip().strip('"')
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None
    return text or None


def explain(score: float, level: str, reasons: list[Reason], use_llm: bool = True) -> Explanation:
    for _ in range(2 if use_llm else 0):
        text = llm_sentence(score, level, reasons)
        if text and is_grounded(text, score, reasons, level):
            return Explanation(score, level, reasons, text, "llm")
    return Explanation(score, level, reasons, template_sentence(score, level, reasons), "template")


# --- queue summary ----------------------------------------------------------

QUEUE_VERBS = {"refund": "to refund", "monitor": "to monitor", "accept": "to approve"}


def queue_facts(count: dict, total: dict, reasons: dict[str, list[str]] | None = None) -> str:
    """The plan as plain facts: the only source of numbers for the summary."""
    lines = [f"Payments waiting for review: {sum(count.values())}"]
    for act in ("refund", "monitor", "accept"):
        lines.append(f"- {_plural(count[act])} {QUEUE_VERBS[act]}, ${total[act]:,.2f} in total")
    lines.append(f"Estimated loss avoided by the refunds: ${total['refund']:,.2f}")
    flagged = [t for texts in (reasons or {}).values() for t in texts]
    if flagged:
        lines.append("Why the flagged payments stand out:")
        lines += [f"- {t}" for t in dict.fromkeys(flagged)]
    return "\n".join(lines)


def queue_template(count: dict, total: dict) -> str:
    return (f"{_plural(count['refund'])} to refund, {count['monitor']} to monitor, "
            f"{count['accept']} to approve. Estimated loss avoided: ${total['refund']:,.2f}.")


def queue_summary(count: dict, total: dict, reasons: dict[str, list[str]] | None = None,
                  use_llm: bool = True) -> tuple[str, str]:
    """(summary, source). Every number the LLM writes must appear in the facts."""
    facts = queue_facts(count, total, reasons)
    if sum(count.values()) == 0:
        return "No payment is waiting for review.", "template"
    allowed = numbers_in(facts)
    prompt = (
        "You are a fraud-review agent reporting to an online seller. In 2 short sentences of "
        "plain English, summarise your plan for the payments waiting for review: how many to "
        "refund, monitor and approve, the estimated loss avoided, and in a few words why the "
        "flagged payments stand out. Use only the facts below. Copy every number and dollar "
        "amount exactly as written, using digits; do not round, add up or compute anything. "
        "Never mention countries or nationalities. No preamble, no lists, no quotes.\n\n" + facts
    )
    for _ in range(2 if use_llm else 0):
        text = llm_complete(prompt, max_tokens=3000)
        if text and numbers_in(text) <= allowed and not names_country(text):
            return text, "llm"
    return queue_template(count, total), "template"


def _plural(n: int) -> str:
    return f"{n} payment" + ("" if n == 1 else "s")


def _facts(score: float, level: str, reasons: list[Reason]) -> str:
    lines = [f"Risk score: {score:.1f} out of 100" + (f" ({level} risk)" if level else "")]
    for r in reasons:
        lines.append(f"- {neutral_text(r)} ({'raises' if r.raises_risk else 'lowers'} the risk)")
    return "\n".join(lines)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _plain(v):
    return v.item() if hasattr(v, "item") else v
