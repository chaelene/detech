"""FastAPI routers for streaming ingress and alert fan-out."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from starlette import status

from app.services.streaming import AlertManager, RateLimitExceeded, StreamingService
from backend.models.schemas import HealthResponse, StreamAnswer, StreamOffer

logger = logging.getLogger(__name__)

router = APIRouter()


def get_streaming_service(request: Request) -> StreamingService:
    service = getattr(request.app.state, "streaming_service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Streaming service unavailable")
    return service


def get_alert_manager(request: Request) -> AlertManager:
    manager = getattr(request.app.state, "alert_manager", None)
    if not manager:
        raise HTTPException(status_code=503, detail="Alert manager unavailable")
    return manager


def get_redis(request: Request):
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return redis


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_endpoint(request: Request, service: StreamingService = Depends(get_streaming_service)) -> HealthResponse:
    redis = get_redis(request)
    redis_ok = False
    try:
        await redis.ping()
        redis_ok = True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Redis ping failed: %s", exc)

    stats = await service.health()
    status_label = "healthy" if redis_ok and stats.get("mqtt") else "degraded"

    return HealthResponse(
        status=status_label,
        redis_available=redis_ok,
        mqtt_connected=bool(stats.get("mqtt")),
        active_sessions=stats.get("active_sessions", []),
    )


@router.post("/stream", response_model=StreamAnswer, status_code=status.HTTP_202_ACCEPTED, tags=["stream"])
async def ingest_stream(
    offer: StreamOffer,
    service: StreamingService = Depends(get_streaming_service),
) -> StreamAnswer:
    logger.info("Received stream offer for session %s", offer.session_id)
    try:
        handle = await service.ingest(offer)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to ingest stream: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to ingest WebRTC stream") from exc

    return StreamAnswer(session_id=offer.session_id, answer_sdp=handle.answer_sdp)


@router.websocket("/alerts")
async def alerts_websocket(websocket: WebSocket) -> None:
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="session_id required")
        return

    redis = getattr(websocket.app.state, "redis", None)
    if redis is None:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="redis unavailable")
        return

    session_data = await redis.hgetall(f"session:{session_id}")
    if not session_data:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unknown session")
        return

    manager: AlertManager = getattr(websocket.app.state, "alert_manager")
    service: StreamingService = getattr(websocket.app.state, "streaming_service")

    await manager.register(session_id, websocket)
    try:
        await manager.wait_for_disconnect(websocket)
    finally:
        await manager.unregister(session_id, websocket)
        await service.handle_disconnect(session_id)

