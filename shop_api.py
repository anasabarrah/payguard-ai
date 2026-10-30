"""PayPal sandbox demo shop and verified webhook receiver."""
from __future__ import annotations

import html
import json
import os

from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient, PayPalError

load_dotenv()

PRODUCTS = {
    "starter": {"name": "Starter kit", "price": 24.00, "description": "PayGuard demo starter kit"},
    "plus": {"name": "Plus bundle", "price": 79.00, "description": "PayGuard demo plus bundle"},
    "pro": {"name": "Pro bundle", "price": 249.00, "description": "PayGuard demo pro bundle"},
}
PRODUCT_JSON = json.dumps(PRODUCTS)


def paypal() -> PayPalClient:
    try:
        return PayPalClient()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def ledger() -> Ledger:
    # One connection per call: these handlers run in worker threads.
    return Ledger()


async def read_json(request: Request) -> dict | None:
    try:
        data = json.loads(await request.body())
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def shop(request):
    client_id = html.escape(os.getenv("PAYPAL_CLIENT_ID", ""), quote=True)
    cards = "".join(
        f'<article><div class="emoji">{icon}</div><h2>{item["name"]}</h2>'
        f'<p>Sandbox demo item · instant checkout</p><strong>${item["price"]:.2f} USD</strong>'
        f'<div id="paypal-{sku}" class="buttons"></div></article>'
        for (sku, item), icon in zip(PRODUCTS.items(), ["🛡️", "✨", "🚀"])
    )
    return HTMLResponse(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>PayGuard Demo Shop</title><script src="https://www.paypal.com/sdk/js?client-id={client_id}&currency=USD&intent=capture"></script>
    <style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fc;color:#14243a;font:16px Inter,system-ui,sans-serif}}header{{background:linear-gradient(120deg,#003087,#0070e0);color:white;padding:42px 7%;}}header h1{{margin:0;font-size:2.4rem}}header p{{opacity:.86}}main{{max-width:1100px;margin:34px auto;padding:0 20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:20px}}article{{background:white;border:1px solid #e2e9f3;border-radius:18px;padding:24px;box-shadow:0 8px 28px #002b5c0d}}.emoji{{font-size:35px}}article h2{{margin:12px 0 4px}}article p{{color:#65758b;min-height:24px}}article strong{{display:block;font-size:1.4rem;margin:18px 0}}.buttons{{min-height:46px}}.note{{color:#65758b;margin:25px 0}}a{{color:#0070e0}}</style></head>
    <body><header><h1>🛡️ PayGuard Demo Shop</h1><p>Pay with your PayPal sandbox buyer account. Orders are captured securely and reviewed by PayGuard AI.</p></header>
    <main><div class="grid">{cards}</div><p class="note">Sandbox only · after approval, the capture appears in <a href="/">the PayGuard dashboard</a> with its risk score.</p></main>
    <script>
    const products = {PRODUCT_JSON};
    for (const [sku, product] of Object.entries(products)) paypal.Buttons({{
      createOrder: async () => {{ const r = await fetch('/api/shop/orders', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{sku}})}}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Could not create order'); return d.id; }},
      onApprove: async (data) => {{ const r = await fetch(`/api/shop/orders/${{data.orderID}}/capture`, {{method:'POST'}}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Capture failed'); alert(`Payment captured: $${{d.amount}} USD · Capture ${{d.capture_id}}`); }},
      onError: e => alert('PayPal checkout error: '+e.message)
    }}).render('#paypal-'+sku);
    </script></body></html>''')


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
                                        description=item["description"], custom_id=sku)
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
        capture_id = ledger().record_capture(captured)
        amount = captured["purchase_units"][0]["payments"]["captures"][0]["amount"]["value"]
        return JSONResponse({"capture_id": capture_id, "amount": amount, "status": "COMPLETED"})
    except PayPalError:
        return JSONResponse({"detail": "PayPal could not capture the order"}, status_code=502)


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
        await run_in_threadpool(ledger().record_webhook, event)
    return JSONResponse({"received": True})


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
