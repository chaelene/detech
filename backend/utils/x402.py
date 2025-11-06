"""Utilities for interacting with Solana x402 payment helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from ..app.models.config import Settings
from ..app.services.streaming import AlertManager

try:  # Optional dependency; backend may run in mock mode without solana libraries
    from shared.solana_utils import SolanaUSDCTransactor, TransferResult, USDCTransferError

    _SOLANA_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - solana stack optional in mock deployments
    SolanaUSDCTransactor = None  # type: ignore[assignment]
    TransferResult = Any  # type: ignore[assignment]

    class USDCTransferError(Exception):
        """Placeholder error when Solana stack is unavailable."""

    _SOLANA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BillingSnapshot:
    """Materialised billing state derived from Redis."""

    balance: Decimal
    last_charge_epoch: float
    total_spent: Decimal
    user_secret: Optional[str]


class X402Handler:
    """Evaluates per-session balances and executes USDC transfers when due."""

    def __init__(
        self,
        *,
        redis,
        settings: Settings,
        transactor: SolanaUSDCTransactor,
    ) -> None:
        self._redis = redis
        self._settings = settings
        self._transactor = transactor
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._charge_amount = Decimal(str(self._settings.x402_charge_per_interval_usdc))
        self._charge_interval = max(1, int(self._settings.x402_charge_interval_seconds))

        if not _SOLANA_AVAILABLE and not self._settings.x402_mock_transfers:
            raise RuntimeError(
                "Solana libraries are missing but x402_mock_transfers is disabled. "
                "Install the solana Python SDK or enable mock mode."
            )

    async def process_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        if not session_id:
            return payload

        async with self._locks[session_id]:
            session_key = f"session:{session_id}"
            session_state = await self._redis.hgetall(session_key)
            if not session_state:
                return payload

            billing = self._materialise_billing_state(session_state)
            now = time.time()
            intervals_due = self._intervals_due(billing.last_charge_epoch, now)
            if intervals_due <= 0:
                return payload

            amount_due = self._charge_amount * intervals_due
            enriched = dict(payload)

            if billing.balance < amount_due:
                enriched["x402"] = {
                    "status": "insufficient_funds",
                    "required": self._format_decimal(amount_due),
                    "balance": self._format_decimal(billing.balance),
                    "intervals": int(intervals_due),
                }
                return enriched

            if not billing.user_secret:
                enriched["x402"] = {
                    "status": "missing_user_secret",
                    "required": self._format_decimal(amount_due),
                }
                return enriched

            reference = enriched.get("alert_id") or f"{session_id}-{int(now)}"

            try:
                result = await self._transactor.transfer_usdc(
                    user_secret=billing.user_secret,
                    amount=amount_due,
                    reference=reference,
                )
            except USDCTransferError as exc:
                logger.error("x402 transfer failed for session %s: %s", session_id, exc)
                enriched["x402"] = {
                    "status": "transfer_failed",
                    "error": str(exc),
                    "required": self._format_decimal(amount_due),
                }
                return enriched

            new_balance = billing.balance - amount_due
            total_spent = billing.total_spent + amount_due
            await self._redis.hset(
                session_key,
                mapping={
                    "x402_balance_usdc": self._format_decimal(new_balance),
                    "x402_last_charge_ts": str(now),
                    "x402_total_spent_usdc": self._format_decimal(total_spent),
                },
            )

            enriched["x402"] = {
                "status": "charged",
                "tx_signature": result.signature,
                "charged_amount": self._format_decimal(amount_due),
                "balance": self._format_decimal(new_balance),
                "attempt": result.attempt,
                "intervals": int(intervals_due),
            }
            enriched["x402_tx_signature"] = result.signature
            logger.info(
                "x402 charged session %s", session_id, extra={"signature": result.signature, "amount": str(amount_due)}
            )
            return enriched

    def _materialise_billing_state(self, session_state: Dict[str, Any]) -> BillingSnapshot:
        balance = self._parse_decimal(session_state.get("x402_balance_usdc", "0"))
        total_spent = self._parse_decimal(session_state.get("x402_total_spent_usdc", "0"))
        last_charge_epoch_raw = session_state.get("x402_last_charge_ts")
        try:
            last_charge_epoch = float(last_charge_epoch_raw) if last_charge_epoch_raw else 0.0
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            last_charge_epoch = 0.0
        user_secret = session_state.get("x402_user_secret")
        return BillingSnapshot(
            balance=balance,
            last_charge_epoch=last_charge_epoch,
            total_spent=total_spent,
            user_secret=user_secret,
        )

    @staticmethod
    def _parse_decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except InvalidOperation:  # pragma: no cover - defensive parsing
            return Decimal("0")

    def _format_decimal(self, amount: Decimal) -> str:
        return format(amount.quantize(Decimal("0.000001")), "f")

    def _intervals_due(self, last_charge_epoch: float, now: float) -> int:
        if last_charge_epoch <= 0:
            return 1
        elapsed = now - last_charge_epoch
        if elapsed < self._charge_interval:
            return 0
        return max(1, int(elapsed // self._charge_interval))


