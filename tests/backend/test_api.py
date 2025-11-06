"""Tests covering the DETECH streaming API surface."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert isinstance(payload["active_sessions"], list)


def test_stream_ingestion_publishes_frames(client: TestClient, mock_mqtt_service):
    session_id = "sess-123"
    response = client.post(
        "/stream",
        json={
            "session_id": session_id,
            "wallet_pubkey": "wallet-test",
            "sdp": "offer-sdp",
        },
    )
    assert response.status_code == 202
    mock_mqtt_service.wait_for_publish(timeout=1)
    assert mock_mqtt_service.published
    assert mock_mqtt_service.published[0]["session_id"] == session_id


def test_stream_rate_limit(client: TestClient):
    payload = {"session_id": "sess-rate", "wallet_pubkey": "wallet-rate", "sdp": "offer"}
    first = client.post("/stream", json=payload)
    assert first.status_code == 202
    second = client.post("/stream", json=payload)
    assert second.status_code == 429


def test_alerts_websocket_broadcast(client: TestClient, mock_mqtt_service, mock_redis):
    session_id = "sess-alert"
    client.post(
        "/stream",
        json={"session_id": session_id, "wallet_pubkey": "wallet-alert", "sdp": "offer"},
    )

    session_bucket = mock_redis._hashes[f"session:{session_id}"]
    session_bucket["x402_balance_usdc"] = "0.01"
    session_bucket["x402_user_secret"] = "dummy-user"
    session_bucket["x402_last_charge_ts"] = str(time.time() - 5)

    with client.websocket_connect(f"/alerts?session_id={session_id}") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        mock_mqtt_service.emit_alert({"session_id": session_id, "label": "person"})
        message = websocket.receive_json()
        assert message["session_id"] == session_id
        assert message["label"] == "person"
        assert message["x402"]["status"] == "insufficient_funds"
