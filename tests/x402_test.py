"""Unit tests for the x402 alert payment handler."""
import time
from decimal import Decimal

import pytest

from app.models.config import Settings
from backend.utils.x402 import X402Handler
from shared.solana_utils import TransferResult


class InMemoryRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)


class StubTransactor:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def transfer_usdc(self, *, user_secret: str, amount: Decimal, reference: str) -> TransferResult:
        self.calls.append({
            "user_secret": user_secret,
            "amount": str(amount),
            "reference": reference,
        })
        return TransferResult(signature="stub-signature", attempt=len(self.calls))

    async def close(self) -> None:  # pragma: no cover - interface parity
        return None


@pytest.mark.asyncio
async def test_x402_handler_processes_payment_when_due():
    redis = InMemoryRedis()
    session_key = "session:alpha"
    redis._hashes[session_key] = {
        "wallet_pubkey": "wallet-alpha",
        "x402_balance_usdc": "1.00",
        "x402_last_charge_ts": str(time.time() - 1.5),
        "x402_user_secret": "example-secret",
        "x402_total_spent_usdc": "0",
    }

    settings = Settings(x402_charge_interval_seconds=1)
    transactor = StubTransactor()
    handler = X402Handler(redis=redis, settings=settings, transactor=transactor)

    payload = {"session_id": "alpha", "alert_id": "alert-1"}
    enriched = await handler.process_alert(payload)

    assert enriched["x402"]["status"] == "charged"
    assert enriched["x402"]["tx_signature"] == "stub-signature"
    assert enriched["x402_tx_signature"] == "stub-signature"
    assert transactor.calls
    updated_balance = Decimal(redis._hashes[session_key]["x402_balance_usdc"])
    assert updated_balance == Decimal("0.95")
    assert Decimal(redis._hashes[session_key]["x402_total_spent_usdc"]) == Decimal("0.05")


@pytest.mark.asyncio
async def test_x402_handler_insufficient_balance_skips_transfer():
    redis = InMemoryRedis()
    session_key = "session:beta"
    redis._hashes[session_key] = {
        "wallet_pubkey": "wallet-beta",
        "x402_balance_usdc": "0.01",
        "x402_last_charge_ts": str(time.time() - 2),
        "x402_user_secret": "example-secret",
        "x402_total_spent_usdc": "0",
    }

    settings = Settings(x402_charge_interval_seconds=1)
    transactor = StubTransactor()
    handler = X402Handler(redis=redis, settings=settings, transactor=transactor)

    payload = {"session_id": "beta", "alert_id": "alert-2"}
    enriched = await handler.process_alert(payload)

    assert enriched["x402"]["status"] == "insufficient_funds"
    assert not transactor.calls
    assert redis._hashes[session_key]["x402_balance_usdc"] == "0.01"


@pytest.mark.asyncio
async def test_x402_handler_missing_secret_marks_payload():
    redis = InMemoryRedis()
    session_key = "session:gamma"
    redis._hashes[session_key] = {
        "wallet_pubkey": "wallet-gamma",
        "x402_balance_usdc": "1.00",
        "x402_last_charge_ts": str(time.time() - 2),
        "x402_total_spent_usdc": "0",
    }

    settings = Settings(x402_charge_interval_seconds=1)
    transactor = StubTransactor()
    handler = X402Handler(redis=redis, settings=settings, transactor=transactor)

    payload = {"session_id": "gamma"}
    enriched = await handler.process_alert(payload)

    assert enriched["x402"]["status"] == "missing_user_secret"
    assert not transactor.calls

