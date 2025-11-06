"""Pytest configuration for backend streaming tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from app.models.config import Settings
from app.services.streaming import MediaSessionHandle
from backend.server import AppDependencies, create_app
from shared.solana_utils import TransferResult


class FakeRedis:
    """Simple in-memory Redis stand-in for tests."""

    def __init__(self) -> None:
        self._hashes: Dict[str, Dict[str, Any]] = {}
        self._counters: Dict[str, int] = {}

    async def ping(self) -> bool:
        return True

    async def incr(self, key: str) -> int:
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    async def expire(self, key: str, ttl: int) -> None:  # pragma: no cover - noop
        return None

    async def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)

    async def hgetall(self, key: str) -> Dict[str, Any]:
        return dict(self._hashes.get(key, {}))

    async def delete(self, key: str) -> None:
        self._hashes.pop(key, None)

    async def close(self) -> None:  # pragma: no cover - noop
        return None

    async def wait_closed(self) -> None:  # pragma: no cover - noop
        return None


class MockMQTTService:
    """Records published frames and simulates alert consumption."""

    def __init__(self) -> None:
        self.alert_handler = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.published: list[Dict[str, Any]] = []
        self._publish_event: asyncio.Event | None = None

    async def start(self, alert_handler):
        self.alert_handler = alert_handler
        self.loop = asyncio.get_running_loop()
        self._publish_event = asyncio.Event()

    async def publish_frame(self, *, session_id: str, wallet_pubkey: str, frame_bytes: bytes) -> None:
        if self._publish_event is None:
            raise RuntimeError("MockMQTTService not started")
        self.published.append(
            {
                "session_id": session_id,
                "wallet_pubkey": wallet_pubkey,
                "frame": frame_bytes,
            }
        )
        self._publish_event.set()

    def wait_for_publish(self, timeout: float = 0.5) -> None:
        if not self.loop or not self._publish_event:
            raise RuntimeError("MockMQTTService not initialised")

        async def _wait():
            await asyncio.wait_for(self._publish_event.wait(), timeout=timeout)
            self._publish_event.clear()

        future = asyncio.run_coroutine_threadsafe(_wait(), self.loop)
        future.result(timeout=timeout)

    def emit_alert(self, payload: Dict[str, Any]) -> None:
        if not self.loop or not self.alert_handler:
            raise RuntimeError("Alert handler not configured")

        future = asyncio.run_coroutine_threadsafe(self.alert_handler(payload), self.loop)
        future.result(timeout=1)

    async def stop(self) -> None:  # pragma: no cover - noop
        return None

    def is_connected(self) -> bool:
        return True


class MockMediasoupController:
    """Provides deterministic session handles for tests."""

    def __init__(self) -> None:
        self.offers: list[Any] = []
        self.cleaned: list[str] = []

    async def accept_offer(self, offer: Any) -> MediaSessionHandle:
        self.offers.append(offer)

        async def frame_provider() -> bytes:
            await asyncio.sleep(0)
            return b"mock-jpeg"

        async def cleanup() -> None:
            self.cleaned.append(offer.session_id)

        return MediaSessionHandle(
            session_id=offer.session_id,
            answer_sdp="mock-answer",
            frame_provider=frame_provider,
            cleanup=cleanup,
        )


class DummyTransactor:
    """Captures transfer invocations without touching the network."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    async def transfer_usdc(self, *, user_secret: str, amount, reference: str):
        self.calls.append({
            "user_secret": user_secret,
            "amount": amount,
            "reference": reference,
        })
        return TransferResult(signature="dummy-signature", attempt=1)

    async def close(self) -> None:  # pragma: no cover - noop
        return None


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def mock_mqtt_service() -> MockMQTTService:
    return MockMQTTService()


@pytest.fixture
def mock_mediasoup_service() -> MockMediasoupController:
    return MockMediasoupController()


@pytest.fixture
def dummy_transactor() -> DummyTransactor:
    return DummyTransactor()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        frame_sample_interval_seconds=0.01,
        rate_limit_max_streams=1,
        cors_origins=["http://localhost:3000"],
        x402_charge_interval_seconds=1,
    )


@pytest.fixture
def app(test_settings, mock_redis, mock_mqtt_service, mock_mediasoup_service, dummy_transactor):
    deps = AppDependencies(
        redis=mock_redis,
        mqtt_service=mock_mqtt_service,
        mediasoup_service=mock_mediasoup_service,
        solana_transactor=dummy_transactor,
    )
    application = create_app(settings=test_settings, dependencies=deps)
    return application


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
