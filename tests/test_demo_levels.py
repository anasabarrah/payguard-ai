"""The demo must show 4 low / 2 medium / 2 high whatever the time of day."""
import random
from collections import Counter
from unittest.mock import MagicMock

import pytest

from payguard.agent import PayGuardAgent
from payguard.ledger import Ledger
from payguard.risk import RiskModel
from payguard.simulate import demo_plan, simulate_payments


def _run(client_ledger, model, day, hour, seed, start):
    led = client_ledger
    plan = demo_plan(random.Random(seed))
    client = MagicMock()
    client.create_order.side_effect = [
        {"id": f"O{start + i}", "status": "COMPLETED", "purchase_units": [{"payments": {
            "captures": [{"id": f"C{start + i}", "status": "COMPLETED",
                          "create_time": f"2026-11-{day:02d}T{hour:02d}:{i:02d}:00Z",
                          "amount": {"currency_code": "USD", "value": f"{a:.2f}"}}]}}]}
        for i, (_, a, _) in enumerate(plan)]
    ids = simulate_payments(client, led, seed=seed)
    alerts = {a.capture_id: a for a in PayGuardAgent(led, model, use_llm=False).review()}
    return Counter(alerts[i].level for i in ids)


@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


@pytest.mark.parametrize("hour", [3, 12, 22])
def test_fresh_ledger(tmp_path, model, hour):
    for seed in range(15):
        led = Ledger(tmp_path / f"{hour}_{seed}.db")
        assert _run(led, model, 2, hour, seed, 0) == {"low": 4, "medium": 2, "high": 2}, seed


def test_repeated_demos_in_one_ledger(tmp_path, model):
    """Back-to-back demos (earlier payments shift the median ticket)."""
    led = Ledger(tmp_path / "busy.db")
    for k, hour in enumerate([3, 7, 12, 16, 22, 23]):
        got = _run(led, model, 3, hour, 100 + k, 100 * k)
        assert got == {"low": 4, "medium": 2, "high": 2}, (hour, got)
