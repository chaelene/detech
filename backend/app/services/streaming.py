"""Streaming service primitives for WebRTC + MQTT integration.

This module centralises the low-latency streaming pipeline: Mediasoup offers are
accepted, frames are sampled and forwarded to Jetson edge consumers over MQTT,
and alert fan-out to WebSocket clients is orchestrated. All helpers are async
first so the FastAPI lifespan hook can manage them cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

import aiomqtt
from aiomqtt import MqttError
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from ..models.config import Settings

logger = logging.getLogger(__name__)


FrameProvider = Callable[[], Awaitable[bytes]]
CleanupCallback = Callable[[], Awaitable[None]]


@dataclass
class MediaSessionHandle:
    """Handle returned by Mediasoup controller for an active session."""

    session_id: str
    answer_sdp: str
    frame_provider: FrameProvider
    cleanup: CleanupCallback


class RateLimitExceeded(Exception):
    """Raised when the rate limiter prevents a new session."""


class RateLimiter:
    """Redis-backed token counter enforcing per-wallet limits."""

    def __init__(self, redis, *, window_seconds: int, max_tokens: int):
        self._redis = redis
        self._window = window_seconds
        self._max_tokens = max_tokens

    async def check(self, wallet_pubkey: str) -> None:
        key = f"rate:{wallet_pubkey}"
        current = await self._redis.incr(key)
        if current == 1:
            await self._redis.expire(key, self._window)
        if current > self._max_tokens:
            logger.warning("Rate limit exceeded for wallet %s", wallet_pubkey)
            raise RateLimitExceeded(f"Too many stream attempts for {wallet_pubkey}")


class MQTTService:
    """Async MQTT helper using aiomqtt for publishing frames and receiving alerts."""

    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        frames_topic: str,
        alerts_topic: str,
        reconnect_interval: float = 2.0,
    ) -> None:
        self._hostname = hostname
        self._port = port
        self._frames_topic = frames_topic
        self._alerts_topic = alerts_topic
        self._reconnect_interval = max(0.5, reconnect_interval)

        self._client: Optional[aiomqtt.Client] = None
        self._connected = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._alert_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    async def start(self, alert_handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Connect to MQTT broker and begin streaming alerts to handler."""

        if self._task and not self._task.done():
            logger.debug("MQTTService already running; start() noop")
            return

        self._alert_handler = alert_handler
        self._stop_event.clear()
        self._task = asyncio.create_task(self._connection_loop(), name="mqtt-connection-loop")

    async def _connection_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to MQTT broker %s:%s", self._hostname, self._port)
                async with aiomqtt.Client(hostname=self._hostname, port=self._port) as client:
                    self._client = client
                    messages_ctx = getattr(client, "messages", None)
                    if callable(messages_ctx):
                        context_manager = messages_ctx()
                    else:
                        context_manager = messages_ctx

                    if context_manager and hasattr(context_manager, "__aenter"):
                        async with context_manager as messages:
                            await client.subscribe(self._alerts_topic)
                            self._connected.set()
                            logger.info("MQTT connected; subscribed to %s", self._alerts_topic)
                            async for message in messages:
                                await self._handle_alert_message(message)
                    else:
                        logger.warning("aiomqtt Client.messages is not an async context manager; falling back to ISO interface")
                        async with client.messages() as messages:  # type: ignore[attr-defined]
                            await client.subscribe(self._alerts_topic)
                            self._connected.set()
                            logger.info("MQTT connected; subscribed to %s", self._alerts_topic)
                            async for message in messages:
                                await self._handle_alert_message(message)
            except asyncio.CancelledError:
                logger.debug("MQTT connection loop cancelled")
                break
            except MqttError as exc:
                self._connected.clear()
                self._client = None
                if self._stop_event.is_set():
                    break
                logger.warning("MQTT connection error: %s", exc)
                await asyncio.sleep(self._reconnect_interval)
            except Exception as exc:  # pragma: no cover - defensive catch
                self._connected.clear()
                self._client = None
                if self._stop_event.is_set():
                    break
                logger.exception("Unexpected MQTT failure: %s", exc)
                await asyncio.sleep(self._reconnect_interval)
            finally:
                self._connected.clear()
                self._client = None

        logger.info("MQTT connection loop terminated")

    async def _handle_alert_message(self, message: Any) -> None:
        if not self._alert_handler:
            return

        try:
            payload = json.loads(message.payload.decode())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Invalid MQTT payload: %s", exc)
            return

        payload.setdefault("received_at", datetime.now(timezone.utc).isoformat())
        try:
            await self._alert_handler(payload)
        except Exception as exc:  # pragma: no cover - downstream issues
            logger.exception("Alert handler raised: %s", exc)

    async def publish_frame(
        self,
        *,
        session_id: str,
        wallet_pubkey: str,
        frame_bytes: bytes,
    ) -> None:
        await self._connected.wait()
        client = self._client
        if client is None:
            raise RuntimeError("MQTT client not started")

        sent_at = datetime.now(timezone.utc)
        frame_b64 = base64.b64encode(frame_bytes).decode()
        payload = {
            "session_id": session_id,
            "wallet_pubkey": wallet_pubkey,
            "timestamp": sent_at.isoformat(),
            "ingest_sent_at": sent_at.isoformat(),
            "frame": frame_b64,
            "frame_jpeg": frame_b64,
        }

        try:
            await client.publish(
                self._frames_topic,
                json.dumps(payload, separators=(",", ":")),
                qos=1,
            )
        except MqttError as exc:
            self._connected.clear()
            logger.warning("Failed to publish frame to MQTT: %s", exc)
            raise

    def is_connected(self) -> bool:
        return self._connected.is_set()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._client = None
        self._connected.clear()


class AlertManager:
    """Tracks WebSocket subscribers and forwards parsed alert payloads."""

    def __init__(self, payment_hook=None, swarm_handler=None) -> None:
        self._connections: Dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._payment_hook = payment_hook
        self._swarm_handler = swarm_handler

    def attach_payment_hook(self, hook) -> None:
        self._payment_hook = hook

    def attach_swarm_handler(self, handler) -> None:
        self._swarm_handler = handler

    async def register(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[session_id].add(websocket)
        await websocket.send_json({"type": "ready", "session_id": session_id})

    async def unregister(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    self._connections.pop(session_id, None)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        message = dict(payload)
        ingest_sent_at = (
            message.get("ingest_sent_at")
            or (message.get("metadata") or {}).get("ingest_sent_at")
            if isinstance(message.get("metadata"), dict)
            else None
        )

        latency_ms: Optional[float] = None
        if ingest_sent_at is not None:
            parsed = self._parse_timestamp(ingest_sent_at)
            if parsed is not None:
                latency_ms = (datetime.now(timezone.utc) - parsed).total_seconds() * 1000.0
                metrics = message.setdefault("metrics", {})
                metrics["ingest_latency_ms"] = round(latency_ms, 2)

        if self._swarm_handler:
            try:
                message = await self._swarm_handler(message)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Swarm handler raised: %s", exc)

        if self._payment_hook:
            try:
                message = await self._payment_hook(message)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("x402 handler raised: %s", exc)
                message = dict(message)
                message.setdefault("x402", {})
                message["x402"].update({"status": "handler_error", "error": str(exc)})

        session_id = message.get("session_id")
        if latency_ms is not None:
            logger.info(
                "Alert delivered",
                extra={
                    "session_id": session_id,
                    "latency_ms": round(latency_ms, 2),
                    "alert_id": message.get("alert_id"),
                    "swarm_confidence": message.get("swarm_confidence"),
                },
            )
        async with self._lock:
            targets = (
                set(self._connections.get(session_id, set()))
                if session_id
                else {ws for peers in self._connections.values() for ws in peers}
            )

        stale: list[tuple[str, WebSocket]] = []
        for websocket in targets:
            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                stale.append((session_id, websocket))
            except RuntimeError as exc:  # pragma: no cover - best effort logging
                logger.warning("Failed to deliver alert: %s", exc)

        for sid, websocket in stale:
            await self.unregister(sid, websocket)

    async def wait_for_disconnect(self, websocket: WebSocket) -> None:
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    async def shutdown(self) -> None:
        """Close active sockets gracefully."""

        async with self._lock:
            connections = list(self._connections.items())
            self._connections.clear()

        for session_id, websockets in connections:
            for websocket in websockets:
                with suppress(Exception):  # pragma: no cover - shutdown best effort
                    await websocket.close(code=1001, reason="Server shutdown")

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    return datetime.fromtimestamp(float(text), tz=timezone.utc)
                except ValueError:
                    return None
        return None


class StreamSession:
    """Represents an active ingestion session and its frame sampling loop."""

    def __init__(
        self,
        *,
        offer: Any,
        handle: MediaSessionHandle,
        mqtt: MQTTService,
        redis,
        settings: Settings,
    ) -> None:
        self.offer = offer
        self.handle = handle
        self.mqtt = mqtt
        self.redis = redis
        self.settings = settings
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        billing_metadata = (self.offer.metadata or {}).get("x402", {}) if getattr(self.offer, "metadata", None) else {}
        balance_value = billing_metadata.get("balance_usdc") if isinstance(billing_metadata, dict) else None
        balance_str = str(balance_value) if balance_value is not None else "0"
        user_secret = billing_metadata.get("user_secret") if isinstance(billing_metadata, dict) else None
        now_ts = datetime.now(timezone.utc).timestamp()

        redis_mapping: Dict[str, Any] = {
            "wallet_pubkey": self.offer.wallet_pubkey,
            "status": "active",
            "metadata": json.dumps(self._metadata_payload()),
            "x402_balance_usdc": balance_str,
            "x402_last_charge_ts": str(now_ts),
            "x402_total_spent_usdc": "0",
        }
        if user_secret:
            redis_mapping["x402_user_secret"] = user_secret

        await self.redis.hset(
            self._redis_key,
            mapping=redis_mapping,
        )
        await self.redis.expire(self._redis_key, self.settings.session_ttl_seconds)

        self._task = asyncio.create_task(self._frame_loop())

    @property
    def _redis_key(self) -> str:
        return f"session:{self.offer.session_id}"

    async def _frame_loop(self) -> None:
        logger.debug("Starting frame loop for session %s", self.offer.session_id)
        try:
            while not self._stopped.is_set():
                frame = await self.handle.frame_provider()
                if frame:
                    await self.mqtt.publish_frame(
                        session_id=self.offer.session_id,
                        wallet_pubkey=self.offer.wallet_pubkey,
                        frame_bytes=frame,
                    )
                await asyncio.sleep(self.settings.frame_sample_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Frame loop error: %s", exc)
        finally:
            await self.redis.hset(self._redis_key, mapping={"status": "stopped"})
            await self.handle.cleanup()
            self._stopped.set()
            logger.debug("Stopped frame loop for session %s", self.offer.session_id)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        await self.redis.delete(self._redis_key)

    def _metadata_payload(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        base_metadata = getattr(self.offer, "metadata", None)
        if base_metadata:
            metadata.update(base_metadata)
        ice_candidates = getattr(self.offer, "ice_candidates", None)
        if ice_candidates:
            metadata["ice_candidates"] = ice_candidates
        return metadata


class StreamingService:
    """Coordinates Mediasoup ingestion, rate limiting, and MQTT egress."""

    def __init__(
        self,
        *,
        mediasoup_controller,
        mqtt_service: MQTTService,
        redis,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.mediasoup_controller = mediasoup_controller
        self.mqtt = mqtt_service
        self.redis = redis
        self.rate_limiter = RateLimiter(
            redis,
            window_seconds=self.settings.rate_limit_window_seconds,
            max_tokens=self.settings.rate_limit_max_streams,
        )
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

    async def ingest(self, offer) -> MediaSessionHandle:
        await self.rate_limiter.check(offer.wallet_pubkey)
        handle: MediaSessionHandle = await self.mediasoup_controller.accept_offer(offer)
        session = StreamSession(
            offer=offer,
            handle=handle,
            mqtt=self.mqtt,
            redis=self.redis,
            settings=self.settings,
        )
        async with self._lock:
            existing = self._sessions.get(offer.session_id)
            if existing:
                await existing.stop()
            await session.start()
            self._sessions[offer.session_id] = session
        return handle

    async def handle_disconnect(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await session.stop()

    async def shutdown(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
        for _, session in sessions:
            await session.stop()

    async def health(self) -> Dict[str, Any]:
        active = list(self._sessions.keys())
        return {
            "redis": True,
            "mqtt": self.mqtt.is_connected(),
            "active_sessions": active,
        }

    def active_session_ids(self) -> list[str]:
        return list(self._sessions.keys())
