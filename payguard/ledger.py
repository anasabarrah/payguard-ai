"""Local SQLite ledger of the payments PayGuard created, captured or refunded.

PayPal's Transaction Search API needs an extra app permission and lags by
hours in sandbox, so the app keeps its own record of every PayPal call.
"""
from __future__ import annotations

import json
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
"""

# Columns added after the first release; created on open if missing.
EXTRA_COLUMNS = {
    "decision": "TEXT",            # accept | monitor | refund; NULL while pending
    "decided_at": "TEXT",
    "explanation": "TEXT",         # cached sentence for explained_score
    "explanation_source": "TEXT",
    "explained_score": "REAL",
}


class Ledger:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(payments)")}
        for name, kind in EXTRA_COLUMNS.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE payments ADD COLUMN {name} {kind}")
        self.conn.commit()

    def record_capture(self, order: dict, payer_id: str | None = None,
                       payer_email: str | None = None, country: str | None = None) -> str:
        """Store a captured order (the response of create/capture_order)."""
        cap = order["purchase_units"][0]["payments"]["captures"][0]
        payer = order.get("payer", {})
        self.conn.execute(
            "INSERT OR REPLACE INTO payments (capture_id, order_id, created_at, amount, currency,"
            " status, payer_id, payer_email, country, raw) VALUES (?,?,?,?,?,?,?,?,?,?)",
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

    def record_refund(self, capture_id: str, refund: dict) -> None:
        value = float(refund["amount"]["value"])
        self.conn.execute(
            "UPDATE payments SET refunded = refunded + ?, status = CASE WHEN refunded + ? >= amount"
            " THEN 'REFUNDED' ELSE 'PARTIALLY_REFUNDED' END WHERE capture_id = ?",
            (value, value, capture_id),
        )
        self.conn.commit()

    def list_payments(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM payments ORDER BY created_at DESC, rowid DESC").fetchall()
        return [{k: r[k] for k in r.keys() if k != "raw"} for r in rows]

    def get(self, capture_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM payments WHERE capture_id = ?", (capture_id,)).fetchone()
        return dict(r) if r else None

    def record_decision(self, capture_id: str, action: str, risk_score: float | None = None,
                        proposed_action: str | None = None, refund_id: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO decisions (capture_id, decided_at, risk_score, proposed_action,"
            " chosen_action, refund_id) VALUES (?,?,?,?,?,?)",
            (capture_id, now, risk_score, proposed_action, action, refund_id),
        )
        self.conn.execute("UPDATE payments SET decision = ?, decided_at = ? WHERE capture_id = ?",
                          (action, now, capture_id))
        self.conn.commit()

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
