"""Integration tests spanning backend, MQTT jetson mock, swarm and payment flow."""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from typing import Any, Dict

import httpx
import pytest


BACKEND_HTTP_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_WS_URL = os.getenv("BACKEND_WS_URL", BACKEND_HTTP_URL.replace("http", "ws"))


def _make_offer_payload(session_id: str, wallet: str) -> Dict[str, Any]:
    dummy_sdp = base64.b64encode(b"v=0\no=- 0 0 IN IP4 127.0.0.1\n").decode()
    return {
        "session_id": session_id,
        "wallet_pubkey": wallet,
        "sdp": dummy_sdp,
        "ice_candidates": [],
        "metadata": {
            "device": "integration-test",
            "x402": {
                "balance_usdc": 5.0,
                "user_secret": "integration-test-secret",
            },
        },
    }


@pytest.mark.asyncio
async def test_stream_to_alert_latency_and_payment() -> None:
    """Verify frame ingestion triggers refined alert with latency metrics and x402 charge."""

    session_id = f"it-{uuid.uuid4().hex[:8]}"
    wallet_pubkey = "TestWallet111111111111111111111111111111111"

    async with httpx.AsyncClient(timeout=30.0) as client:
        offer_payload = _make_offer_payload(session_id, wallet_pubkey)
        response = await client.post(f"{BACKEND_HTTP_URL}/stream", json=offer_payload)
        assert response.status_code == 202, response.text

        async with client.ws_connect(f"{BACKEND_WS_URL}/alerts?session_id={session_id}") as websocket:
            ready_message = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            assert ready_message.get("type") == "ready"

            alert_message = await asyncio.wait_for(websocket.receive_json(), timeout=45)

        metrics = alert_message.get("metrics", {})
        assert metrics, "Alert payload missing metrics"
        latency_ms = metrics.get("ingest_latency_ms")
        assert isinstance(latency_ms, (int, float)) and latency_ms > 0, "Latency metric not recorded"
        assert latency_ms < 5000, f"Latency too high: {latency_ms}ms"

        swarm = alert_message.get("swarm") or {}
        refined = swarm.get("refined") or {}
        accuracy = refined.get("accuracy")
        assert accuracy is not None and accuracy >= 0.7, f"Swarm accuracy not elevated: {accuracy}"
        evolution = swarm.get("evolution") or {}
        delta = evolution.get("delta")
        assert delta is not None and delta > 0, "Swarm evolution delta should be positive"

        assert alert_message.get("swarm_confidence", 0) >= 0.7

        x402 = alert_message.get("x402") or {}
        assert x402.get("status") == "charged", f"x402 did not charge: {x402}"
        assert "tx_signature" in x402, "x402 charge should include mock transaction signature"
