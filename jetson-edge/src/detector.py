"""Asynchronous MQTT detector service for DETECH Jetson edge devices.

This module ingests base64 encoded frames from the `detech/frames` topic,
executes the perception pipeline (YOLO11 + MediaPipe gestures), tracks objects
with a Kalman filter, and publishes enriched alerts to `detech/alerts`.

The service keeps raw detections on-device and only publishes minimal metadata
required for downstream vigilance, enabling low-latency edge intelligence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import paho.mqtt.client as mqtt

from utils.frame_processor import FrameDecodeError, FrameProcessor


try:  # Optional dependency to hydrate environment variables from .env files
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


LOGGER = logging.getLogger("detector")


def _parse_port(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None

    if port <= 0 or port >= 65536:
        return None
    return port


def _load_environment() -> None:
    if load_dotenv is None:
        return

    search_paths = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    )

    for candidate in search_paths:
        if candidate.is_file():
            load_dotenv(candidate)  # type: ignore[misc]
            return

    load_dotenv()  # type: ignore[misc]


def _resolve_broker_defaults() -> Tuple[str, int]:
    broker_host = os.environ.get("MQTT_BROKER_HOST")
    broker_port = _parse_port(os.environ.get("MQTT_BROKER_PORT"))

    broker_url = os.environ.get("MQTT_BROKER_URL")
    if broker_url:
        parsed = urlparse(broker_url if "://" in broker_url else f"mqtt://{broker_url}")
        broker_host = parsed.hostname or broker_host
        if parsed.port:
            broker_port = parsed.port

    return broker_host or "localhost", broker_port or 1883


_load_environment()
DEFAULT_BROKER_HOST, DEFAULT_BROKER_PORT = _resolve_broker_defaults()


class DetectorService:
    """Asynchronous MQTT subscriber -> detector -> publisher pipeline."""

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        frames_topic: str,
        alerts_topic: str,
        client_id: Optional[str] = None,
        processor: Optional[FrameProcessor] = None,
        queue_size: int = 4,
        reconnect_interval: float = 2.0,
        connect_timeout: float = 10.0,
        max_connect_attempts: int = 5,
        max_backoff: float = 30.0,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.frames_topic = frames_topic
        self.alerts_topic = alerts_topic
        self.client_id = client_id or f"detech-jetson-{socket.gethostname()}"
        self.processor = processor or FrameProcessor()
        self.queue: asyncio.Queue[Tuple[str, bytes]] = asyncio.Queue(maxsize=max(1, queue_size))
        self.reconnect_interval = max(0.5, reconnect_interval)
        self.connect_timeout = max(1.0, connect_timeout)
        self.max_connect_attempts = max(1, max_connect_attempts)
        self.max_backoff = max(self.reconnect_interval, max_backoff)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = asyncio.Event()
        self._connected = asyncio.Event()
        self._network_loop_running = False
        self._firewall_checked = False

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.enable_logger(logging.getLogger("paho"))
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        min_delay = max(1, int(round(self.reconnect_interval)))
        max_delay = max(min_delay, int(round(self.max_backoff)))
        self.client.reconnect_delay_set(min_delay=min_delay, max_delay=max_delay)

    # ------------------------------------------------------------------
    # MQTT callbacks (executed in paho network thread)
    # ------------------------------------------------------------------

    def _on_connect(self, client: mqtt.Client, userdata, flags, rc):  # type: ignore[override]
        if rc == mqtt.CONNACK_ACCEPTED:
            LOGGER.info("Connected to MQTT broker %s:%s", self.broker_host, self.broker_port)
            client.subscribe(self.frames_topic, qos=1)
            self._connected.set()
        else:
            LOGGER.error("MQTT connection failed with code %s", rc)

    def _on_disconnect(self, client: mqtt.Client, userdata, rc):  # type: ignore[override]
        LOGGER.warning("Disconnected from MQTT broker (code=%s)", rc)
        self._connected.clear()

    def _on_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):  # type: ignore[override]
        if not self._loop or self._stop_event.is_set():
            return

        payload = msg.payload

        async def _enqueue() -> None:
            if self.queue.full():
                # Drop the oldest frame to favour recency
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            await self.queue.put((msg.topic, payload))

        try:
            asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)
        except RuntimeError:
            LOGGER.debug("Event loop not available; dropping incoming frame")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()

        await self._connect()

        LOGGER.info(
            "Detector service listening | frames_topic=%s | alerts_topic=%s", self.frames_topic, self.alerts_topic
        )

        loop = self._loop
        assert loop is not None

        try:
            while not self._stop_event.is_set():
                try:
                    topic, payload = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                start = loop.time()
                alert = await self._process_message(topic, payload)
                self.queue.task_done()

                if alert:
                    await self._publish_alert(alert)

                elapsed = loop.time() - start
                latency_ms = elapsed * 1000.0
                LOGGER.debug("End-to-end latency %.1f ms | topic=%s", latency_ms, topic)
                if latency_ms > 500:
                    LOGGER.warning("Latency threshold exceeded: %.1f ms (>500 ms)", latency_ms)
                target_interval = max(0.0, (1.0 / 15.0) - elapsed)
                if target_interval > 0:
                    await asyncio.sleep(target_interval)
        finally:
            await self._disconnect()

    async def stop(self) -> None:
        if not self._stop_event.is_set():
            self._stop_event.set()

    async def _connect(self) -> None:
        await self._ensure_firewall_allows_port()

        attempt = 1
        backoff = self.reconnect_interval
        last_exc: Optional[BaseException] = None

        while attempt <= self.max_connect_attempts and not self._stop_event.is_set():
            LOGGER.info(
                "Connecting to MQTT broker %s:%s (attempt %s/%s)",
                self.broker_host,
                self.broker_port,
                attempt,
                self.max_connect_attempts,
            )

            self._connected.clear()

            try:
                self.client.connect_async(self.broker_host, self.broker_port, keepalive=30)
                self._start_network_loop()
                await asyncio.wait_for(self._connected.wait(), timeout=self.connect_timeout)
                LOGGER.debug(
                    "Established MQTT connection to %s:%s after %s attempt(s)",
                    self.broker_host,
                    self.broker_port,
                    attempt,
                )
                return
            except asyncio.TimeoutError as exc:
                last_exc = exc
                LOGGER.warning(
                    "MQTT connection attempt %s timed out after %.1fs",
                    attempt,
                    self.connect_timeout,
                )
            except Exception as exc:  # pragma: no cover - unexpected network errors
                last_exc = exc
                LOGGER.exception("MQTT connection attempt %s failed", attempt)
            finally:
                if not self._connected.is_set():
                    self._stop_network_loop(force=True)
                    try:
                        self.client.disconnect()
                    except Exception:  # pragma: no cover - best effort cleanup
                        LOGGER.debug("MQTT disconnect cleanup failed", exc_info=True)

            if attempt >= self.max_connect_attempts or self._stop_event.is_set():
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)
            attempt += 1

        raise ConnectionError(
            f"Failed to connect to MQTT broker {self.broker_host}:{self.broker_port} "
            f"after {self.max_connect_attempts} attempts"
        ) from last_exc

    async def _disconnect(self) -> None:
        LOGGER.info("Shutting down detector service")
        try:
            self._stop_network_loop(force=True)
            try:
                self.client.disconnect()
            except Exception:  # pragma: no cover - best effort cleanup
                LOGGER.debug("Ignoring MQTT disconnect error during shutdown", exc_info=True)
        finally:
            self.processor.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_network_loop(self) -> None:
        if not self._network_loop_running:
            self.client.loop_start()
            self._network_loop_running = True

    def _stop_network_loop(self, *, force: bool = False) -> None:
        if self._network_loop_running:
            self.client.loop_stop(force=force)
            self._network_loop_running = False

    async def _ensure_firewall_allows_port(self) -> None:
        if self._firewall_checked:
            return

        self._firewall_checked = True

        if not sys.platform.startswith("linux"):
            LOGGER.debug("Skipping firewall check on non-Linux platform (%s)", sys.platform)
            return

        ufw_path = shutil.which("ufw")
        if not ufw_path:
            LOGGER.debug("UFW not installed; skipping MQTT firewall validation")
            return

        def _query_ufw() -> Tuple[bool, str]:
            try:
                proc = subprocess.run(
                    [ufw_path, "status"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:  # pragma: no cover - best effort guard
                return False, str(exc)

            output = proc.stdout or proc.stderr or ""
            return proc.returncode == 0, output

        success, output = await asyncio.to_thread(_query_ufw)
        if not success:
            LOGGER.debug("Unable to query UFW status: %s", output)
            return

        text = output.lower()
        if "inactive" in text:
            LOGGER.debug("UFW inactive; no MQTT firewall adjustment required")
            return

        if str(self.broker_port) not in text:
            LOGGER.warning(
                "UFW active and MQTT port %s is not explicitly allowed. "
                "Run `sudo ufw allow %s/tcp` on the Jetson to permit MQTT traffic.",
                self.broker_port,
                self.broker_port,
            )

    async def _process_message(self, topic: str, payload: bytes) -> Optional[Dict[str, Any]]:
        frame_b64, metadata = self._extract_frame_payload(payload)
        if not frame_b64:
            return None

        timestamp = self._resolve_timestamp(metadata)
        result = await asyncio.to_thread(self.processor.process_base64_frame, frame_b64, timestamp)
        if not result:
            return None

        if metadata:
            meta_slot = result.get("metadata")
            if not isinstance(meta_slot, dict):
                meta_slot = {}
            meta_slot.update({k: v for k, v in metadata.items() if k != "frame"})
            result["metadata"] = meta_slot
        elif not result.get("metadata"):
            result.pop("metadata", None)
        result["source"] = self.client_id
        return result

    async def _publish_alert(self, alert: Dict[str, Any]) -> None:
        payload = json.dumps(alert, ensure_ascii=True, separators=(",", ":"))
        info = self.client.publish(self.alerts_topic, payload=payload, qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.error("Failed to publish alert: rc=%s", info.rc)

    def _extract_frame_payload(self, payload: bytes) -> Tuple[Optional[str], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}

        if not payload:
            return None, metadata

        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            return None, metadata

        frame_b64: Optional[str] = None

        if text.startswith("{") and text.endswith("}"):
            try:
                message = json.loads(text)
                frame_b64 = message.get("frame") or message.get("data")
                metadata = {k: v for k, v in message.items() if k not in {"frame", "data"}}
            except json.JSONDecodeError:
                LOGGER.debug("Failed to parse JSON payload; assuming raw base64")
                frame_b64 = text
        else:
            frame_b64 = text

        return frame_b64, metadata

    def _resolve_timestamp(self, metadata: Dict[str, Any]) -> Optional[float]:
        value = metadata.get("timestamp")
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            ts_text = value.rstrip("Z")
            try:
                dt = datetime.fromisoformat(ts_text)
                return dt.timestamp()
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    LOGGER.debug("Unable to parse timestamp: %s", value)
        return None


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DETECH Jetson MQTT detector service")
    parser.add_argument("--broker-host", default=DEFAULT_BROKER_HOST, help="MQTT broker hostname or IP")
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT, help="MQTT broker port")
    parser.add_argument("--frames-topic", default="detech/frames", help="Topic to subscribe for base64 frames")
    parser.add_argument("--alerts-topic", default="detech/alerts", help="Topic to publish enriched alerts")
    parser.add_argument("--client-id", default=None, help="Optional MQTT client ID")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO11 weights")
    parser.add_argument("--engine", default=None, help="Path to optional TensorRT engine")
    parser.add_argument("--verbose", action="count", default=0, help="Increase logging verbosity")
    return parser.parse_args(argv)


async def _run_service(args: argparse.Namespace) -> None:
    configure_logging(args.verbose)

    processor = FrameProcessor(model_path=args.model, engine_path=args.engine)
    service = DetectorService(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        frames_topic=args.frames_topic,
        alerts_topic=args.alerts_topic,
        client_id=args.client_id,
        processor=processor,
    )

    async def _shutdown_handler() -> None:
        await service.stop()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown_handler()))

    try:
        await service.run()
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received; stopping service")
        await service.stop()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(_run_service(args))
    except FrameDecodeError as exc:
        LOGGER.error("Frame decoding error: %s", exc)
    except ConnectionError as exc:
        LOGGER.error("MQTT connection error: %s", exc)


if __name__ == "__main__":
    main()

