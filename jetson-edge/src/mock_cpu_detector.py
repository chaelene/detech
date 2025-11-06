"""Lightweight MQTT detector used in docker-compose CPU simulations.

This service replaces the GPU-heavy perception stack with a deterministic
publisher that still exercises the end-to-end pipeline:

- Subscribes to the frame topic (default: ``detech/frames``)
- Parses base64 frame envelopes emitted by the backend
- Emits enriched alerts on the alerts topic (default: ``detech/alerts``)

The goal is to provide predictable payloads for integration tests while keeping
latency metrics and swarm enrichment fields wired through the system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt


LOGGER = logging.getLogger("mock-cpu-detector")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MockCpuDetector:
    """Minimal MQTT worker that responds to frame events with canned alerts."""

    def __init__(
        self,
        *,
        broker_host: str,
        broker_port: int,
        frames_topic: str,
        alerts_topic: str,
        reconnect_interval: float = 2.0,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.frames_topic = frames_topic
        self.alerts_topic = alerts_topic
        self.reconnect_interval = reconnect_interval

        client_id = f"detech-mock-{socket.gethostname()}"
        self.client = mqtt.Client(client_id=client_id)
        self.client.enable_logger(logging.getLogger("paho.mock"))
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._queue: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue(maxsize=8)

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client: mqtt.Client, userdata, flags, rc):  # type: ignore[override]
        if rc == mqtt.CONNACK_ACCEPTED:
            LOGGER.info("Connected to MQTT broker %s:%s", self.broker_host, self.broker_port)
            client.subscribe(self.frames_topic, qos=1)
            self._connected.set()
        else:
            LOGGER.error("MQTT connection failed: rc=%s", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata, rc):  # type: ignore[override]
        if rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.warning("Disconnected from MQTT broker (rc=%s)", rc)
        self._connected.clear()

    def _on_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):  # type: ignore[override]
        if not self._loop or self._stopping.is_set():
            return

        async def _enqueue() -> None:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put((msg.topic, msg.payload))

        try:
            asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)
        except RuntimeError:
            LOGGER.debug("Event loop unavailable; dropping frame payload")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._connect()
        LOGGER.info(
            "Mock CPU detector ready | frames_topic=%s | alerts_topic=%s",
            self.frames_topic,
            self.alerts_topic,
        )

        while not self._stopping.is_set():
            try:
                topic, payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            alert = await asyncio.to_thread(self._build_alert, topic, payload)
            self._queue.task_done()

            if alert:
                await self._publish_alert(alert)

        await self._disconnect()

    async def stop(self) -> None:
        self._stopping.set()

    async def _connect(self) -> None:
        LOGGER.info("Connecting to MQTT broker %s:%s", self.broker_host, self.broker_port)
        self.client.connect_async(self.broker_host, self.broker_port, keepalive=45)
        self.client.loop_start()
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=10)
        except asyncio.TimeoutError as exc:
            raise ConnectionError("Timed out waiting for MQTT connection") from exc

    async def _disconnect(self) -> None:
        LOGGER.info("Stopping mock detector")
        self.client.loop_stop()
        self.client.disconnect()

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def _build_alert(self, topic: str, payload: bytes) -> Optional[Dict[str, Any]]:
        if topic != self.frames_topic:
            LOGGER.debug("Ignoring payload for topic %s", topic)
            return None

        frame, metadata = self._deserialize_frame(payload)
        if metadata is None:
            return None

        session_id = metadata.get("session_id") or f"session-{uuid.uuid4()}"
        ingest_sent_at = metadata.get("ingest_sent_at")
        wallet = metadata.get("wallet_pubkey")
        base_timestamp = metadata.get("timestamp")
        if isinstance(base_timestamp, str):
            try:
                ts = datetime.fromisoformat(base_timestamp.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = time.time()
        else:
            ts = _safe_float(base_timestamp, default=time.time())

        detection_ts = time.time()
        detections = [
            {
                "name": "person",
                "confidence": round(random.uniform(0.78, 0.92), 4),
                "bbox": [random.randint(100, 220), random.randint(120, 200), 160, 220],
            }
        ]

        anomaly_score = round(random.uniform(0.6, 0.85), 4)
        swarm_accuracy = round(random.uniform(0.74, 0.88), 4)

        alert_payload: Dict[str, Any] = {
            "alert_id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": detection_ts,
            "type": "edge_detection",
            "severity": "medium",
            "description": "Mock CPU detector classified potential anomaly",
            "source": "jetson",
            "wallet_pubkey": wallet,
            "ingest_sent_at": ingest_sent_at,
            "detection": {
                "timestamp": detection_ts,
                "objects": detections,
                "gestures": [],
                "ingest_frame_timestamp": ts,
            },
            "swarm": {
                "refined": {
                    "accuracy": swarm_accuracy,
                    "threat_level": "elevated" if anomaly_score > 0.7 else "observed",
                    "anomaly_score": anomaly_score,
                    "confidence": swarm_accuracy,
                },
                "commands": ["dispatch-drone"] if anomaly_score > 0.75 else ["log-event"],
            },
            "swarm_confidence": swarm_accuracy,
            "anomaly_score": anomaly_score,
            "metadata": {
                "ingest_sent_at": ingest_sent_at,
                "ingest_frame_length": len(frame or b""),
                "received_at": _now_iso(),
                "session_id": session_id,
                "wallet_pubkey": wallet,
            },
        }

        return alert_payload

    def _deserialize_frame(self, payload: bytes) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        if not payload:
            return None, None

        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload, {}

        text = text.strip()
        if not text:
            return None, None

        if text.startswith("{") and text.endswith("}"):
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                return payload, {}

            metadata = dict(message)
            frame_b64 = metadata.pop("frame", None) or metadata.pop("frame_jpeg", None)
            if frame_b64:
                try:
                    frame_bytes = frame_b64.encode("utf-8")
                except AttributeError:
                    frame_bytes = None
            else:
                frame_bytes = None

            # bubble up known metadata fields
            for key in ("session_id", "wallet_pubkey", "timestamp", "ingest_sent_at"):
                if key not in metadata and key in message:
                    metadata[key] = message[key]

            return frame_bytes, metadata

        return payload, {}

    async def _publish_alert(self, alert: Dict[str, Any]) -> None:
        data = json.dumps(alert, separators=(",", ":"))
        info = self.client.publish(self.alerts_topic, payload=data, qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.error("Failed to publish alert: rc=%s", info.rc)


async def _run_service() -> None:
    logging.basicConfig(
        level=os.environ.get("DETECH_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    broker_host = os.environ.get("MQTT_BROKER_HOST", "localhost")
    broker_port = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
    frames_topic = os.environ.get("FRAMES_TOPIC", "detech/frames")
    alerts_topic = os.environ.get("ALERTS_TOPIC", "detech/alerts")

    detector = MockCpuDetector(
        broker_host=broker_host,
        broker_port=broker_port,
        frames_topic=frames_topic,
        alerts_topic=alerts_topic,
    )

    loop = asyncio.get_running_loop()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(detector.stop()))

    try:
        await detector.run()
    except KeyboardInterrupt:
        LOGGER.info("Mock detector interrupted")
    finally:
        await detector.stop()


def main() -> None:
    asyncio.run(_run_service())


if __name__ == "__main__":
    main()

