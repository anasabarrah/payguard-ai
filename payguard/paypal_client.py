"""Minimal PayPal REST v2 client for the sandbox environment.

Credentials are read from environment variables (loaded from .env):
PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, and optionally PAYPAL_BASE_URL.
"""
from __future__ import annotations

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

    def _request(self, method: str, path: str, *, json: dict | None = None,
                 params: dict | None = None, idempotent: bool = False,
                 headers: dict | None = None) -> dict:
        h = {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }
        if idempotent:
            h["PayPal-Request-Id"] = str(uuid.uuid4())
        if headers:
            h.update(headers)
        resp = self.session.request(method, f"{self.base_url}{path}", json=json,
                                    params=params, headers=h, timeout=self.timeout)
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

    def verify_webhook_signature(self, webhook_id: str, headers: dict,
                                 event: dict) -> bool:
        """Ask PayPal to verify the signature on a webhook event."""
        transmission = {
            "auth_algo": headers.get("paypal-auth-algo"),
            "cert_url": headers.get("paypal-cert-url"),
            "transmission_id": headers.get("paypal-transmission-id"),
            "transmission_sig": headers.get("paypal-transmission-sig"),
            "transmission_time": headers.get("paypal-transmission-time"),
            "webhook_id": webhook_id,
            "webhook_event": event,
        }
        if not all(transmission[k] for k in ("auth_algo", "cert_url", "transmission_id",
                                              "transmission_sig", "transmission_time")):
            return False
        result = self._request("POST", "/v1/notifications/verify-webhook-signature",
                               json=transmission)
        return result.get("verification_status") == "SUCCESS"

    def list_webhooks(self) -> list[dict]:
        return self._request("GET", "/v1/notifications/webhooks").get("webhooks", [])

    def create_webhook(self, url: str, events: list[str]) -> dict:
        return self._request("POST", "/v1/notifications/webhooks", json={
            "url": url, "event_types": [{"name": name} for name in events],
        }, idempotent=True)

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


def capture_id_of(captured_order: dict) -> str:
    return captured_order["purchase_units"][0]["payments"]["captures"][0]["id"]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _body(resp: requests.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return resp.text
