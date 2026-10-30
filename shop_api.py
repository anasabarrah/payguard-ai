"""PayPal sandbox demo shop and verified webhook receiver."""
from __future__ import annotations

import html
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from payguard.agent import PayGuardAgent
from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient, PayPalError
from payguard.risk import RiskModel

load_dotenv()

# The server's catalog is the only source of prices: the browser sends a SKU, never an amount.
PRODUCTS = {
    "bottle": {"name": "Insulated bottle", "price": 24.00, "art": "bottle",
               "blurb": "Keeps water cold for 24 hours. 750 ml, steel.",
               "tint": "#e8f1fd", "ink": "#0070e0"},
    "beanie": {"name": "Merino beanie", "price": 39.00, "art": "beanie",
               "blurb": "Soft, warm, and it does not itch.",
               "tint": "#fdf2e8", "ink": "#b54708"},
    "pack": {"name": "Daypack 22 L", "price": 89.00, "art": "pack",
             "blurb": "Light hiking pack with a laptop sleeve.",
             "tint": "#e9f7ef", "ink": "#067647"},
    "tent": {"name": "Two-person tent", "price": 249.00, "art": "tent",
             "blurb": "1.6 kg, sets up in three minutes.",
             "tint": "#f1ecfb", "ink": "#5925dc"},
}
SHOP_HTML = (Path(__file__).resolve().parent / "templates" / "shop.html").read_text()


def paypal() -> PayPalClient:
    try:
        return PayPalClient()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


@lru_cache(maxsize=1)
def risk_model() -> RiskModel:
    return RiskModel.load()


def ledger() -> Ledger:
    # One connection per call: these handlers run in worker threads.
    return Ledger()


async def read_json(request: Request) -> dict | None:
    try:
        data = json.loads(await request.body())
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def judge_box() -> str:
    """Login of the sandbox buyer account set aside for judges (JUDGE_BUYER_* env vars)."""
    email, password = os.getenv("JUDGE_BUYER_EMAIL"), os.getenv("JUDGE_BUYER_PASSWORD")
    warning = ('<p class="warn">⚠️ PayPal <b>sandbox</b> test account only: no real money, no real '
               'card. Never type a real PayPal login on this demo.</p>')
    if not (email and password):
        return ('<section class="judge"><h3>Hackathon judges: test buyer account</h3>'
                '<p style="margin:0">Use the sandbox buyer login given in the project README, '
                'or any PayPal sandbox personal account.</p>' + warning + '</section>')
    return ('<section class="judge"><h3>Hackathon judges: log in to PayPal with this test buyer'
            '</h3><dl><dt>Email</dt><dd>' + html.escape(email) + '</dd><dt>Password</dt><dd>'
            + html.escape(password) + '</dd></dl>' + warning + '</section>')


async def shop(request):
    client_id = os.getenv("PAYPAL_CLIENT_ID", "")
    catalog = json.dumps(PRODUCTS).replace("</", "<\\/")
    page = (SHOP_HTML.replace("__PAYPAL_CLIENT_ID__", quote(client_id, safe=""))
            .replace("__JUDGE_BOX__", judge_box())
            .replace("__PRODUCTS_JSON__", catalog))
    return HTMLResponse(page)


async def products(request):
    return JSONResponse(PRODUCTS)


async def create_order(request: Request):
    data = await read_json(request) or {}
    sku = data.get("sku")
    if sku not in PRODUCTS:
        return JSONResponse({"detail": "Unknown product"}, status_code=400)
    item = PRODUCTS[sku]
    try:
        order = await run_in_threadpool(paypal().create_order, item["price"],
                                        description=f"PayGuard Demo Shop: {item['name']}", custom_id=sku)
    except PayPalError:
        return JSONResponse({"detail": "PayPal could not create the order"}, status_code=502)
    return JSONResponse({"id": order["id"]})


async def capture_order(request: Request):
    return await run_in_threadpool(_capture, request.path_params["order_id"])


def _capture(order_id: str) -> JSONResponse:
    client = paypal()
    try:
        order = client.get_order(order_id)
        units = order.get("purchase_units", [])
        sku = units[0].get("custom_id") if units else None
        if sku not in PRODUCTS:
            return JSONResponse({"detail": "Order is not a PayGuard shop order"}, status_code=400)
        expected = f"{PRODUCTS[sku]['price']:.2f}"
        amount = units[0].get("amount") or {}
        if amount.get("currency_code") != "USD" or amount.get("value") != expected:
            return JSONResponse({"detail": "Order amount does not match the shop catalog"}, status_code=400)
        if order.get("status") == "COMPLETED":
            captured = order
        else:
            captured = client.capture_order(order_id)
        led = ledger()
        capture_id = led.record_capture(captured)
        amount = captured["purchase_units"][0]["payments"]["captures"][0]["amount"]["value"]
    except PayPalError:
        return JSONResponse({"detail": "PayPal could not capture the order"}, status_code=502)
    alert = next(a for a in PayGuardAgent(led, risk_model()).review() if a.capture_id == capture_id)
    return JSONResponse({"capture_id": capture_id, "amount": amount, "status": "COMPLETED",
                         "score": alert.score, "level": alert.level})


async def paypal_webhook(request: Request):
    """Receive a PayPal webhook; anything PayPal does not vouch for is refused."""
    raw = await request.body()
    try:
        event = json.loads(raw)
    except ValueError:
        return JSONResponse({"detail": "Body is not JSON"}, status_code=400)
    if not isinstance(event, dict):
        return JSONResponse({"detail": "Body is not a webhook event"}, status_code=400)
    webhook_id = os.getenv("PAYPAL_WEBHOOK_ID")
    if not webhook_id:
        return JSONResponse({"detail": "PAYPAL_WEBHOOK_ID is not configured"}, status_code=503)
    headers = {k.lower(): v for k, v in request.headers.items()}
    try:
        valid = await run_in_threadpool(paypal().verify_webhook_signature, webhook_id, headers, raw)
    except PayPalError:
        return JSONResponse({"detail": "Could not verify PayPal webhook"}, status_code=502)
    if not valid:
        return JSONResponse({"detail": "Invalid PayPal webhook signature"}, status_code=400)
    if event.get("event_type") in {"PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED"}:
        await run_in_threadpool(_apply_event, event)
    return JSONResponse({"received": True})


def _apply_event(event: dict) -> None:
    led = ledger()
    if not led.record_webhook(event) or event["event_type"] != "PAYMENT.CAPTURE.COMPLETED":
        return
    # A capture PayGuard did not make itself (e.g. its shop capture call never returned):
    # the event has no payer, so fetch the order for the payer's email and country.
    capture_id = (event.get("resource") or {}).get("id")
    row = led.get(capture_id) if capture_id else None
    if row and not row["payer_email"] and row["order_id"] != "UNKNOWN":
        try:
            order = paypal().get_order(row["order_id"])
        except PayPalError:
            return
        captures = [c for u in order.get("purchase_units", [])
                    for c in (u.get("payments") or {}).get("captures", [])]
        if any(c.get("id") == capture_id for c in captures):
            payer = order.get("payer") or {}
            led.record_capture({"id": order["id"], "purchase_units": [{"payments": {
                "captures": [c for c in captures if c["id"] == capture_id]}}], "payer": payer})


async def shop_redirect(request):
    return RedirectResponse("/shop/")


routes = [
    Route("/shop", shop_redirect),
    Route("/shop/", shop),
    Route("/api/shop/products", products),
    Route("/api/shop/orders", create_order, methods=["POST"]),
    Route("/api/shop/orders/{order_id:str}/capture", capture_order, methods=["POST"]),
    Route("/webhooks/paypal", paypal_webhook, methods=["POST"]),
]
