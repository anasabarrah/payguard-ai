"""Local SQLite ledger of the payments PayGuard created, captured or refunded.

PayPal's Transaction Search API needs an extra app permission and lags by
hours in sandbox, so the app keeps its own record of every PayPal call.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    capture_id    TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    amount        REAL NOT NULL,
    currency      TEXT NOT NULL,
    status        TEXT NOT NULL,
    payer_id      TEXT,
    payer_email   TEXT,
    country       TEXT,
    refunded      REAL NOT NULL DEFAULT 0,
    raw           TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id      TEXT NOT NULL REFERENCES payments(capture_id),
    decided_at      TEXT NOT NULL,
    risk_score      REAL,
    proposed_action TEXT,
    chosen_action   TEXT NOT NULL,
    refund_id       TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id   TEXT PRIMARY KEY,
    capture_id  TEXT NOT NULL,
    amount      REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paypal_webhook_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    received_at TEXT NOT NULL
);
"""

# Columns added after the first release; created on open if missing.
EXTRA_COLUMNS = {
    "decision": "TEXT",            # accept | monitor | refund; NULL while pending
    "decided_at": "TEXT",
    "explanation": "TEXT",         # cached sentence for explained_score
    "explanation_source": "TEXT",
    "explained_score": "REAL",
    "webhook_verified_at": "TEXT",  # when PayPal confirmed the capture by a verified webhook
}


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("PAYGUARD_DB") or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(payments)")}
        for name, kind in EXTRA_COLUMNS.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE payments ADD COLUMN {name} {kind}")
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(decisions)")}
        if "source" not in cols:  # who decided: seller | agent_plan | paypal
            self.conn.execute("ALTER TABLE decisions ADD COLUMN source TEXT")
        self.conn.commit()

    def record_capture(self, order: dict, payer_id: str | None = None,
                       payer_email: str | None = None, country: str | None = None) -> str:
        """Store a captured order (the response of create/capture_order)."""
        cap = order["purchase_units"][0]["payments"]["captures"][0]
        payer = order.get("payer", {})
        # Upsert: the same capture can arrive again (its webhook, a retried capture call);
        # keep the seller's decision and any payer details already known.
        self.conn.execute(
            "INSERT INTO payments (capture_id, order_id, created_at, amount, currency,"
            " status, payer_id, payer_email, country, raw) VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(capture_id) DO UPDATE SET order_id = excluded.order_id,"
            " payer_id = COALESCE(excluded.payer_id, payer_id),"
            " payer_email = COALESCE(excluded.payer_email, payer_email),"
            " country = COALESCE(excluded.country, country), raw = excluded.raw",
            (
                cap["id"], order["id"],
                cap.get("create_time") or datetime.now(timezone.utc).isoformat(),
                float(cap["amount"]["value"]), cap["amount"]["currency_code"], cap["status"],
                payer_id or payer.get("payer_id"),
                payer_email or payer.get("email_address"),
                country or payer.get("address", {}).get("country_code"),
                json.dumps(order),
            ),
        )
        self.conn.commit()
        return cap["id"]

    def record_webhook(self, event: dict) -> bool:
        """Apply a verified PayPal capture/refund event once; return whether it was new."""
        event_id, kind = event.get("id"), event.get("event_type")
        resource = event.get("resource") or {}
        if not event_id or kind not in {"PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED"}:
            return False
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO paypal_webhook_events VALUES (?,?,?)",
                (event_id, kind, datetime.now(timezone.utc).isoformat()))
            if not cur.rowcount:
                return False
            if kind == "PAYMENT.CAPTURE.COMPLETED":
                capture_id = resource.get("id")
                amount = resource.get("amount") or {}
                if capture_id and amount.get("value") and amount.get("currency_code"):
                    order_id = ((resource.get("supplementary_data") or {}).get("related_ids") or {}).get("order_id") or "UNKNOWN"
                    now = datetime.now(timezone.utc).isoformat()
                    self.conn.execute(
                        "INSERT OR IGNORE INTO payments (capture_id,order_id,created_at,amount,currency,status,raw)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (capture_id, order_id, resource.get("create_time") or now,
                         float(amount["value"]), amount["currency_code"], resource.get("status", "COMPLETED"),
                         json.dumps(event)))
                    self.conn.execute("UPDATE payments SET webhook_verified_at = ? WHERE capture_id = ?",
                                      (now, capture_id))
            else:
                amount = resource.get("amount") or {}
                related = ((resource.get("supplementary_data") or {}).get("related_ids") or {})
                capture_id = related.get("capture_id")
                if not capture_id:
                    for link in resource.get("links", []):
                        if link.get("rel") == "up":
                            capture_id = link.get("href", "").rstrip("/").split("/")[-1]
                            break
                if capture_id and resource.get("id") and amount.get("value"):
                    self._apply_refund(capture_id, resource["id"], float(amount["value"]))
                    # Refunded from PayPal itself, not from PayGuard: log it so it leaves the queue.
                    row = self.conn.execute("SELECT decision, refunded, amount FROM payments"
                                            " WHERE capture_id = ?", (capture_id,)).fetchone()
                    if row and row["decision"] is None and row["refunded"] >= row["amount"]:
                        self._log_decision(capture_id, "refund", None, None, resource["id"], "paypal")
        return True

    def record_refund(self, capture_id: str, refund: dict) -> None:
        with self.conn:
            self._apply_refund(capture_id, refund["id"], float(refund["amount"]["value"]))

    def _apply_refund(self, capture_id: str, refund_id: str, value: float) -> None:
        """Add a refund once per PayPal refund id (the API response and its webhook both report it)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO refunds VALUES (?,?,?,?)",
            (refund_id, capture_id, value, datetime.now(timezone.utc).isoformat()))
        if not cur.rowcount:
            return
        self.conn.execute(
            "UPDATE payments SET refunded = MIN(amount, refunded + ?), status = CASE"
            " WHEN refunded + ? >= amount THEN 'REFUNDED' ELSE 'PARTIALLY_REFUNDED' END"
            " WHERE capture_id = ?",
            (value, value, capture_id),
        )

    def version(self) -> tuple:
        """Cheap fingerprint of the ledger; it changes whenever a payment, refund or decision does."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(refunded), 0), COUNT(decision), COUNT(webhook_verified_at),"
            " (SELECT COUNT(*) FROM decisions) FROM payments").fetchone()
        return tuple(row)

    def list_payments(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM payments ORDER BY created_at DESC, rowid DESC").fetchall()
        return [{k: r[k] for k in r.keys() if k != "raw"} for r in rows]

    def get(self, capture_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM payments WHERE capture_id = ?", (capture_id,)).fetchone()
        return dict(r) if r else None

    def record_decision(self, capture_id: str, action: str, risk_score: float | None = None,
                        proposed_action: str | None = None, refund_id: str | None = None,
                        source: str = "seller") -> None:
        with self.conn:
            self._log_decision(capture_id, action, risk_score, proposed_action, refund_id, source)

    def _log_decision(self, capture_id: str, action: str, risk_score: float | None,
                      proposed_action: str | None, refund_id: str | None, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO decisions (capture_id, decided_at, risk_score, proposed_action,"
            " chosen_action, refund_id, source) VALUES (?,?,?,?,?,?,?)",
            (capture_id, now, risk_score, proposed_action, action, refund_id, source),
        )
        self.conn.execute("UPDATE payments SET decision = ?, decided_at = ? WHERE capture_id = ?",
                          (action, now, capture_id))

    def decisions(self, capture_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM decisions", ()
        if capture_id:
            q, args = q + " WHERE capture_id = ?", (capture_id,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY id", args).fetchall()]

    def cache_explanation(self, capture_id: str, score: float, sentence: str, source: str) -> None:
        self.conn.execute("UPDATE payments SET explanation = ?, explanation_source = ?,"
                          " explained_score = ? WHERE capture_id = ?",
                          (sentence, source, score, capture_id))
        self.conn.commit()
