"""Pydantic schemas for DETECH backend streaming endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StreamOffer(BaseModel):
    """Offer payload pushed by the frontend WebRTC client."""

    session_id: str = Field(..., description="Client-generated session identifier")
    wallet_pubkey: str = Field(..., description="Solana wallet public key for the user")
    sdp: str = Field(..., description="Base64-encoded SDP offer")
    ice_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional ICE candidates for trickle ICE support",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional client context (device info, resolution, etc.)",
    )


class StreamAnswer(BaseModel):
    """Answer returned after Mediasoup ingest has been accepted."""

    session_id: str
    answer_sdp: str


class HealthResponse(BaseModel):
    """Structured health response for probes."""

    status: str
    redis_available: bool
    mqtt_connected: bool
    active_sessions: List[str]


class AlertMessage(BaseModel):
    """Detection alert parsed from MQTT broker."""

    session_id: str
    label: str
    confidence: Optional[float] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class SessionState(BaseModel):
    """Materialised state stored in Redis for an active session."""

    session_id: str
    wallet_pubkey: str
    status: str
    metadata: Optional[Dict[str, Any]] = None

