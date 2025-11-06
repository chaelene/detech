"""Utility helpers for orchestrating Solana USDC transfers."""

from __future__ import annotations

import asyncio
import base58
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Optional

import uuid

from nacl.signing import SigningKey
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solana.transaction import Transaction
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    TransferCheckedParams,
    create_associated_token_account,
    get_associated_token_address,
    transfer_checked,
)

logger = logging.getLogger(__name__)


USDC_MINT_MAINNET = "EPjFWdd5AufqSSqeM2qoi9GAJh9zn7wo7GSaJ6y4wDR"
USDC_DECIMALS = 6

_TEST_AGENT_SEED = bytes.fromhex("7c" * 32)
_TEST_SIGNING_KEY = SigningKey(_TEST_AGENT_SEED)
_TEST_AGENT_SECRET_BYTES = _TEST_SIGNING_KEY.encode() + _TEST_SIGNING_KEY.verify_key.encode()
DEFAULT_TEST_AGENT_SECRET = base58.b58encode(_TEST_AGENT_SECRET_BYTES).decode()


class USDCTransferError(Exception):
    """Raised when a USDC transfer fails across all retry attempts."""


@dataclass
class TransferResult:
    """Metadata returned after a successful transfer."""

    signature: str
    attempt: int


def _load_keypair(secret: str) -> Keypair:
    try:
        secret_bytes = base58.b58decode(secret)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise USDCTransferError("invalid base58 secret key") from exc
    return Keypair.from_secret_key(secret_bytes)


def _to_raw_amount(amount: Decimal) -> int:
    try:
        quantised = amount.quantize(Decimal("1." + "0" * USDC_DECIMALS), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:  # pragma: no cover - defensive guard
        raise USDCTransferError("invalid amount precision for USDC") from exc
    scaled = quantised * (10 ** USDC_DECIMALS)
    return int(scaled)


class SolanaUSDCTransactor:
    """Thin async wrapper around solana-py for sending USDC transfers with retries."""

    def __init__(
        self,
        *,
        rpc_url: str,
        agent_secret: Optional[str] = None,
        usdc_mint: str = USDC_MINT_MAINNET,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._client = AsyncClient(rpc_url, commitment=Confirmed)
        self._agent = _load_keypair(agent_secret or DEFAULT_TEST_AGENT_SECRET)
        self._usdc_mint = PublicKey(usdc_mint)
        self._max_retries = max(1, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)
        self._agent_ata: Optional[PublicKey] = None

    @property
    def agent_pubkey(self) -> PublicKey:
        return self._agent.pubkey()

    async def close(self) -> None:
        await self._client.close()

    async def transfer_usdc(
        self,
        *,
        user_secret: str,
        amount: Decimal,
        reference: Optional[str] = None,
    ) -> TransferResult:
        payer = _load_keypair(user_secret)
        source_owner = payer.pubkey()
        destination_owner = self._agent.pubkey()

        source_token = await self._ensure_token_account(owner=source_owner, payer=payer)
        destination_token = await self._ensure_agent_token_account()

        raw_amount = _to_raw_amount(amount)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                transaction = await self._build_transfer_transaction(
                    payer=payer,
                    source_token=source_token,
                    destination_token=destination_token,
                    raw_amount=raw_amount,
                )
                response = await self._client.send_transaction(
                    transaction,
                    payer,
                    opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed),
                )
                signature = response.value
                if not isinstance(signature, str):  # pragma: no cover - defensive
                    signature = str(signature)
                logger.info(
                    "x402 transfer success", extra={"signature": signature, "attempt": attempt, "reference": reference}
                )
                return TransferResult(signature=signature, attempt=attempt)
            except Exception as exc:  # pragma: no cover - network errors are environment-specific
                last_exc = exc
                logger.warning("USDC transfer attempt %s failed: %s", attempt, exc)
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._backoff_seconds * (2 ** (attempt - 1)))

        raise USDCTransferError(str(last_exc))

    async def _ensure_agent_token_account(self) -> PublicKey:
        if self._agent_ata is None:
            self._agent_ata = await self._ensure_token_account(owner=self._agent.pubkey(), payer=self._agent)
        return self._agent_ata

    async def _ensure_token_account(self, *, owner: PublicKey, payer: Keypair) -> PublicKey:
        ata = get_associated_token_address(owner, self._usdc_mint)
        info = await self._client.get_account_info(ata)
        if info.value is None:
            transaction = Transaction()
            transaction.add(
                create_associated_token_account(
                    payer=payer.pubkey(),
                    owner=owner,
                    mint=self._usdc_mint,
                )
            )
            latest_blockhash = await self._client.get_latest_blockhash()
            transaction.recent_blockhash = latest_blockhash.value.blockhash
            transaction.fee_payer = payer.pubkey()
            transaction.sign(payer)
            await self._client.send_transaction(
                transaction,
                payer,
                opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed),
            )
        return ata

    async def _build_transfer_transaction(
        self,
        *,
        payer: Keypair,
        source_token: PublicKey,
        destination_token: PublicKey,
        raw_amount: int,
    ) -> Transaction:
        params = TransferCheckedParams(
            program_id=TOKEN_PROGRAM_ID,
            source=source_token,
            mint=self._usdc_mint,
            dest=destination_token,
            authority=payer.pubkey(),
            amount=raw_amount,
            decimals=USDC_DECIMALS,
            signers=[],
        )
        transaction = Transaction()
        transaction.add(transfer_checked(params))
        latest_blockhash = await self._client.get_latest_blockhash()
        transaction.recent_blockhash = latest_blockhash.value.blockhash
        transaction.fee_payer = payer.pubkey()
        transaction.sign(payer)
        return transaction


class MockSolanaTransactor:
    """Drop-in mock that simulates successful USDC transfers without network access."""

    def __init__(self, *, latency_seconds: float = 0.05) -> None:
        self._latency = max(0.0, latency_seconds)

    async def close(self) -> None:
        return None

    async def transfer_usdc(
        self,
        *,
        user_secret: str,
        amount: Decimal,
        reference: Optional[str] = None,
    ) -> TransferResult:
        del user_secret  # Mock implementation does not require the user secret
        await asyncio.sleep(self._latency)
        signature = f"MOCK-{uuid.uuid4().hex[:16]}"
        logger.info(
            "Mock x402 transfer emitted",
            extra={"signature": signature, "reference": reference, "amount": str(amount)},
        )
        return TransferResult(signature=signature, attempt=1)


