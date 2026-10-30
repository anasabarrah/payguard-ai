"""The PayGuard agent: scores payments, explains them, proposes an action.

The agent only *proposes*. Nothing reaches PayPal until the seller confirms a
decision through `execute(..., seller_confirmed=True)`, which is what the UI's
Approve / Monitor / Refund buttons call.
"""
from __future__ import annotations

from dataclasses import dataclass

from payguard.explain import Explanation, Reason, explain, top_reasons
from payguard.features import features_from_payments
from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient
from payguard.risk import RiskModel

ACTIONS = ("accept", "monitor", "refund")
PROPOSAL = {"low": "accept", "medium": "monitor", "high": "refund"}
WHY = {
    "accept": "Low risk: nothing unusual compared with your other payments.",
    "monitor": "Medium risk: keep the order on hold or watch for a chargeback before shipping.",
    "refund": "High risk: refunding now avoids shipping goods and a likely chargeback.",
}


@dataclass
class Alert:
    payment: dict
    score: float
    level: str
    reasons: list[Reason]
    proposed_action: str

    @property
    def capture_id(self) -> str:
        return self.payment["capture_id"]

    @property
    def pending(self) -> bool:
        return self.payment.get("decision") is None

    @property
    def proposal_rationale(self) -> str:
        return WHY[self.proposed_action]


class PayGuardAgent:
    def __init__(self, ledger: Ledger, model: RiskModel, paypal: PayPalClient | None = None,
                 use_llm: bool = True):
        self.ledger = ledger
        self.model = model
        self.paypal = paypal
        self.use_llm = use_llm

    def review(self) -> list[Alert]:
        """Score every payment in the ledger, newest first."""
        payments = self.ledger.list_payments()
        if not payments:
            return []
        feats = features_from_payments(payments)
        scores = self.model.score(feats)
        reasons = top_reasons(self.model, feats)
        alerts = []
        for p, s, r in zip(payments, scores, reasons):
            level = self.model.level(s)
            alerts.append(Alert(p, float(s), level, r, PROPOSAL[level]))
        return alerts

    def explain(self, alert: Alert) -> Explanation:
        """Plain-English sentence for an alert, cached in the ledger per score."""
        p = alert.payment
        if p.get("explanation") and p.get("explained_score") == alert.score:
            return Explanation(alert.score, alert.level, alert.reasons,
                               p["explanation"], p["explanation_source"])
        e = explain(alert.score, alert.level, alert.reasons, use_llm=self.use_llm)
        self.ledger.cache_explanation(alert.capture_id, alert.score, e.sentence, e.source)
        p.update(explanation=e.sentence, explanation_source=e.source, explained_score=alert.score)
        return e

    def execute(self, alert: Alert, action: str, seller_confirmed: bool = False) -> dict:
        """Apply the seller's decision. Refunds call PayPal; the others are recorded locally."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if not seller_confirmed:
            raise PermissionError("the seller must confirm before PayGuard acts")
        current = self.ledger.get(alert.capture_id)
        if current is None:
            raise KeyError(alert.capture_id)
        if current["decision"] == "refund" and action != "refund":
            raise ValueError("payment already refunded; this cannot be undone")

        refund_id = None
        if action == "refund":
            remaining = round(current["amount"] - current["refunded"], 2)
            if remaining <= 0:
                raise ValueError("payment already fully refunded")
            if self.paypal is None:
                raise RuntimeError("PayPal client not configured")
            refund = self.paypal.refund_capture(
                alert.capture_id, amount=remaining, currency=current["currency"],
                note="Refunded by the seller after a PayGuard fraud review")
            self.ledger.record_refund(alert.capture_id, refund)
            refund_id = refund["id"]

        self.ledger.record_decision(alert.capture_id, action, risk_score=alert.score,
                                    proposed_action=alert.proposed_action, refund_id=refund_id)
        return {"capture_id": alert.capture_id, "action": action, "refund_id": refund_id,
                "followed_proposal": action == alert.proposed_action}
