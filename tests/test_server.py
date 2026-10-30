"""Start the app as the Dockerfile does and check that one port serves everything."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "PAYGUARD_DB": str(tmp_path_factory.mktemp("srv") / "ledger.db"),
           "PAYPAL_CLIENT_ID": "test-client", "PAYPAL_CLIENT_SECRET": "test-secret",
           "PAYPAL_BASE_URL": "http://127.0.0.1:9", "PAYPAL_WEBHOOK_ID": ""}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "server.py", "--server.port", str(port),
         "--server.address", "127.0.0.1", "--server.headless", "true"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(f"{base}/_stcore/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def test_one_port_serves_dashboard_shop_and_webhook(server):
    assert requests.get(f"{server}/_stcore/health").text == "ok"
    assert "<div id=\"root\">" in requests.get(f"{server}/").text           # Streamlit SPA
    shop = requests.get(f"{server}/shop/")
    assert shop.ok and "PayGuard Demo Shop" in shop.text
    # Not configured: the webhook refuses instead of trusting the event.
    hook = requests.post(f"{server}/webhooks/paypal", json={"id": "EV", "event_type": "X"})
    assert hook.status_code == 503
