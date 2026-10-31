"""Minimal PayPal REST v2 client for the sandbox environment.

Credentials are read from environment variables (loaded from .env):
PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, and optionally PAYPAL_BASE_URL.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

SANDBOX_URL = "https://api-m.sandbox.paypal.com"
REPORTING_SCOPE = "https://uri.paypal.com/services/reporting/search/read"


class PayPalError(RuntimeError):
    def __init__(self, status: int, body: object):
        super().__init__(f"PayPal API error {status}: {body}")
        self.status = status
        self.body = body

    @property
    def issue(self) -> str:
        """PayPal's machine-readable reason, e.g. REFUND_FAILED_INSUFFICIENT_FUNDS."""
        body = self.body if isinstance(self.body, dict) else {}
        detail = (body.get("details") or [{}])[0]
        return detail.get("issue") or body.get("name") or f"HTTP_{self.status}"

    @property
    def short(self) -> str:
        """One readable line, e.g. 'REFUND_FAILED_INSUFFICIENT_FUNDS: Capture could not be ...'."""
        body = self.body if isinstance(self.body, dict) else {}
        detail = (body.get("details") or [{}])[0]
        issue = detail.get("issue") or body.get("name") or f"HTTP {self.status}"
        text = detail.get("description") or body.get("message") or body.get("error_description") or ""
        return f"{issue}: {text}" if text else issue


class PayPalClient:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None,
                 base_url: str | None = None, timeout: float = 30.0):
        load_dotenv()
        self.client_id = client_id or os.getenv("PAYPAL_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PAYPAL_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise ValueError("PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET must be set in .env")
        self.base_url = (base_url or os.getenv("PAYPAL_BASE_URL") or SANDBOX_URL).rstrip("/")
        self.timeout = timeout
        self._token: str | None = None
        self._token_expiry = 0.0
        self.scopes: set[str] = set()
        self.session = requests.Session()

    # --- auth -------------------------------------------------------------
    def get_access_token(self, force: bool = False) -> str:
        if not force and self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = self.session.post(
            f"{self.base_url}/v1/oauth2/token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise PayPalError(resp.status_code, _body(resp))
        data = resp.json()
        self._token = data["access_token"]
        self.scopes = set(data.get("scope", "").split())
        self._token_expiry = time.time() + int(data.get("expires_in", 0))
        return self._token

    def _request(self, method: str, path: str, *, json: dict | list | None = None,
                 data: str | None = None, params: dict | None = None, idempotent: bool = False,
                 headers: dict | None = None) -> dict:
        h = {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }
        if idempotent:
            h["PayPal-Request-Id"] = str(uuid.uuid4())
        if headers:
            h.update(headers)
        try:
            resp = self.session.request(method, f"{self.base_url}{path}", json=json,
                                        data=data, params=params, headers=h, timeout=self.timeout)
        except requests.RequestException as e:
            raise PayPalError(503, {"name": "PAYPAL_UNREACHABLE", "message": str(e)}) from e
        if resp.status_code >= 400:
            raise PayPalError(resp.status_code, _body(resp))
        return resp.json() if resp.content else {}

    # --- orders -----------------------------------------------------------
    def create_order(self, amount: float | str, currency: str = "USD",
                     description: str | None = None, payment_source: dict | None = None,
                     custom_id: str | None = None) -> dict:
        unit = {"amount": {"currency_code": currency, "value": f"{float(amount):.2f}"}}
        if description:
            unit["description"] = description
        if custom_id:
            unit["custom_id"] = custom_id
        body: dict = {"intent": "CAPTURE", "purchase_units": [unit]}
        if payment_source:
            body["payment_source"] = payment_source
        return self._request("POST", "/v2/checkout/orders", json=body, idempotent=True,
                             headers={"Prefer": "return=representation"})

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/v2/checkout/orders/{order_id}")

    def capture_order(self, order_id: str) -> dict:
        return self._request("POST", f"/v2/checkout/orders/{order_id}/capture", json={},
                             idempotent=True, headers={"Prefer": "return=representation"})

    # --- payments ---------------------------------------------------------
    def get_capture(self, capture_id: str) -> dict:
        return self._request("GET", f"/v2/payments/captures/{capture_id}")

    def refund_capture(self, capture_id: str, amount: float | str | None = None,
                       currency: str = "USD", note: str | None = None) -> dict:
        body: dict = {}
        if amount is not None:
            body["amount"] = {"currency_code": currency, "value": f"{float(amount):.2f}"}
        if note:
            body["note_to_payer"] = note
        return self._request("POST", f"/v2/payments/captures/{capture_id}/refund", json=body,
                             idempotent=True, headers={"Prefer": "return=representation"})

    def verify_webhook_signature(self, webhook_id: str, headers: dict, raw_body: bytes | str) -> bool:
        """Ask PayPal whether a webhook delivery is authentic.

        The event is sent back exactly as received: PayPal signs the raw body, so
        re-serialising the parsed JSON could change it and fail verification.
        """
        fields = {
            "auth_algo": headers.get("paypal-auth-algo"),
            "cert_url": headers.get("paypal-cert-url"),
            "transmission_id": headers.get("paypal-transmission-id"),
            "transmission_sig": headers.get("paypal-transmission-sig"),
            "transmission_time": headers.get("paypal-transmission-time"),
            "webhook_id": webhook_id,
        }
        if not all(fields.values()):
            return False
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        body = json.dumps(fields)[:-1] + ', "webhook_event": ' + raw_body + "}"
        result = self._request("POST", "/v1/notifications/verify-webhook-signature", data=body)
        return result.get("verification_status") == "SUCCESS"

    def list_webhooks(self) -> list[dict]:
        return self._request("GET", "/v1/notifications/webhooks").get("webhooks", [])

    def create_webhook(self, url: str, events: list[str]) -> dict:
        return self._request("POST", "/v1/notifications/webhooks", json={
            "url": url, "event_types": [{"name": name} for name in events],
        }, idempotent=True)

    def set_webhook_events(self, webhook_id: str, events: list[str]) -> dict:
        return self._request("PATCH", f"/v1/notifications/webhooks/{webhook_id}", json=[{
            "op": "replace", "path": "/event_types", "value": [{"name": n} for n in events],
        }])

    # --- disputes (v1) ------------------------------------------------------
    def list_disputes(self, page_size: int = 20) -> list[dict]:
        """Disputes of this seller, newest first, with a summary of each."""
        return self._request("GET", "/v1/customer/disputes",
                             params={"page_size": page_size}).get("items", [])

    def get_dispute(self, dispute_id: str) -> dict:
        return self._request("GET", f"/v1/customer/disputes/{dispute_id}")

    def send_dispute_message(self, dispute_id: str, message: str) -> dict:
        """Message the buyer. PayPal allows it only in the INQUIRY stage."""
        return self._request("POST", f"/v1/customer/disputes/{dispute_id}/send-message",
                             json={"message": message})

    def provide_dispute_evidence(self, dispute_id: str, evidences: list[dict]) -> dict:
        """Submit evidence (multipart, JSON in the `input` part). Only while the dispute is
        WAITING_FOR_SELLER_RESPONSE."""
        payload = json.dumps({"evidences": evidences})
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/customer/disputes/{dispute_id}/provide-evidence",
                headers={"Authorization": f"Bearer {self.get_access_token()}"},
                files={"input": ("input.json", payload, "application/json")},
                timeout=self.timeout)
        except requests.RequestException as e:
            raise PayPalError(503, {"name": "PAYPAL_UNREACHABLE", "message": str(e)}) from e
        if resp.status_code >= 400:
            raise PayPalError(resp.status_code, _body(resp))
        return resp.json() if resp.content else {}

    def require_dispute_evidence(self, dispute_id: str, action: str = "SELLER_EVIDENCE") -> dict:
        """Sandbox only: move an UNDER_REVIEW dispute to WAITING_FOR_SELLER/BUYER_RESPONSE."""
        return self._request("POST", f"/v1/customer/disputes/{dispute_id}/require-evidence",
                             json={"action": action})

    def adjudicate_dispute(self, dispute_id: str, outcome: str) -> dict:
        """Sandbox only: settle an UNDER_REVIEW dispute (BUYER_FAVOR or SELLER_FAVOR)."""
        return self._request("POST", f"/v1/customer/disputes/{dispute_id}/adjudicate",
                             json={"adjudication_outcome": outcome})

    # --- reporting --------------------------------------------------------
    def can_search_transactions(self) -> bool:
        """True once 'Transaction search' is enabled for the app and reflected in the token."""
        self.get_access_token()
        return REPORTING_SCOPE in self.scopes

    def list_transactions(self, start: datetime | None = None, end: datetime | None = None,
                          page_size: int = 100) -> list[dict]:
        """Transaction Search API (v1). Note: sandbox data can lag by up to a few hours."""
        end = end or datetime.now(timezone.utc)
        start = start or end - timedelta(days=30)
        out, page = [], 1
        while True:
            data = self._request("GET", "/v1/reporting/transactions", params={
                "start_date": _iso(start), "end_date": _iso(end), "fields": "all",
                "page_size": page_size, "page": page,
            })
            out.extend(data.get("transaction_details", []))
            if page >= data.get("total_pages", 1):
                return out
            page += 1


REFUND_MESSAGES = {
    "REFUND_FAILED_INSUFFICIENT_FUNDS":
        "The demo seller's PayPal sandbox balance is too low for this refund right now.",
    "CAPTURE_FULLY_REFUNDED": "PayPal says it has already been fully refunded.",
    "REFUND_AMOUNT_EXCEEDED": "PayPal says it has already been fully refunded.",
    "REFUND_TIME_LIMIT_EXCEEDED": "It is too old to be refunded through PayPal.",
    "REFUND_NOT_ALLOWED": "PayPal does not allow this refund.",
    "PERMISSION_DENIED": "The demo seller's PayPal sandbox account is not allowed to refund right now.",
    "PAYPAL_UNREACHABLE": "PayPal is not responding right now. Please try again in a minute.",
}


def refund_failure_message(error: PayPalError) -> str:
    """A polite, non-technical sentence for a refund PayPal refused."""
    reason = REFUND_MESSAGES.get(error.issue)
    if reason is None:
        reason = ("PayPal is having trouble right now. Please try again in a minute."
                  if error.status >= 500 else "PayPal declined the refund.")
    return f"{reason} Nothing was refunded, and it stays in your review queue."


def capture_id_of(captured_order: dict) -> str:
    return captured_order["purchase_units"][0]["payments"]["captures"][0]["id"]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _body(resp: requests.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return resp.text
