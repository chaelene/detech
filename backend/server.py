"""FastAPI application assembly for DETECH backend streaming stack."""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from .app.models.config import Settings
from .app.services.streaming import (
    AlertManager,
    MediaSessionHandle,
    MQTTService,
    StreamingService,
)
from .app.services.swarm import SwarmCoordinator
from backend.utils.x402 import X402Handler
from shared.solana_utils import MockSolanaTransactor, SolanaUSDCTransactor
from backend.routes.stream import router as stream_router

logger = logging.getLogger(__name__)


PLACEHOLDER_JPEG = base64.b64decode(
    (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGy0kJiQnKz0x"
        "NDQ0NzxERUQkPz4+QUVNRzMzR0tNU09bYm5ra2tERj/2wBDARESEhgVGC8dGy9HR0dHR0dH"
        "R0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0dHR0f/wAARCAABAAEDARE"
        "ASIAAhEBAxEB/8QAFwABAQEBAAAAAAAAAAAAAAAAAAIDBf/EABgBAQEBAQEAAAAAAAAAAAAA"
        "AAAAAEQID/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oA"
        "DAMBAAIRAxEAPwD3gA//2Q=="
    )
)


class DefaultMediasoupController:
    """Fallback mediasoup controller for local development and tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def accept_offer(self, offer: Any) -> MediaSessionHandle:
        async def frame_provider() -> bytes:
            return PLACEHOLDER_JPEG

        async def cleanup() -> None:
            return None

        answer = "\n".join(
            [
                "v=0",
                "o=- 0 0 IN IP4 127.0.0.1",
                "s=DETECH Edge Answer",
                "t=0 0",
                "a=ice-lite",
            ]
        )

        return MediaSessionHandle(
            session_id=offer.session_id,
            answer_sdp=answer,
            frame_provider=frame_provider,
            cleanup=cleanup,
        )


@dataclass
class AppDependencies:
    redis: Optional[Any] = None
    mqtt_service: Optional[MQTTService] = None
    mediasoup_service: Optional[Any] = None
    alert_manager: Optional[AlertManager] = None
    streaming_service: Optional[StreamingService] = None
    solana_transactor: Optional[SolanaUSDCTransactor] = None
    x402_handler: Optional[X402Handler] = None
    swarm_coordinator: Optional[SwarmCoordinator] = None


def _create_redis(settings: Settings) -> Redis:
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def create_app(
    *, settings: Optional[Settings] = None, dependencies: Optional[AppDependencies] = None
) -> FastAPI:
    settings = settings or Settings()
    dependencies = dependencies or AppDependencies()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.dependencies = dependencies

        redis_provided = dependencies.redis is not None
        redis = dependencies.redis or _create_redis(settings)
        app.state.redis = redis

        transactor_provided = dependencies.solana_transactor is not None
        if dependencies.solana_transactor is not None:
            solana_transactor = dependencies.solana_transactor
        elif settings.x402_mock_transfers:
            solana_transactor = MockSolanaTransactor()
        else:
            solana_transactor = SolanaUSDCTransactor(
                rpc_url=settings.solana_rpc_url,
                agent_secret=settings.x402_agent_private_key or None,
                usdc_mint=settings.x402_usdc_mint,
                max_retries=settings.x402_max_retries,
                backoff_seconds=settings.x402_retry_backoff_seconds,
            )
        app.state.solana_transactor = solana_transactor

        x402_handler = dependencies.x402_handler or X402Handler(
            redis=redis,
            settings=settings,
            transactor=solana_transactor,
        )
        app.state.x402_handler = x402_handler

        swarm_coordinator = dependencies.swarm_coordinator or SwarmCoordinator(
            redis=redis,
            mock_mode=settings.swarm_mock_mode,
        )
        app.state.swarm_coordinator = swarm_coordinator

        alert_manager = dependencies.alert_manager or AlertManager(
            swarm_handler=swarm_coordinator.process_alert
        )
        alert_manager.attach_payment_hook(x402_handler.process_alert)
        app.state.alert_manager = alert_manager

        mqtt_service = dependencies.mqtt_service or MQTTService(
            hostname=settings.mqtt_broker_host,
            port=settings.mqtt_broker_port,
            frames_topic=settings.mqtt_topic_frames,
            alerts_topic=settings.mqtt_topic_alerts,
            reconnect_interval=settings.mqtt_reconnect_interval_seconds,
        )
        app.state.mqtt_service = mqtt_service

        mediasoup_controller = dependencies.mediasoup_service or DefaultMediasoupController(settings)
        app.state.mediasoup_controller = mediasoup_controller

        streaming_service = dependencies.streaming_service or StreamingService(
            mediasoup_controller=mediasoup_controller,
            mqtt_service=mqtt_service,
            redis=redis,
            settings=settings,
        )
        app.state.streaming_service = streaming_service

        try:
            await redis.ping()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Redis ping failed during startup: %s", exc)

        await mqtt_service.start(alert_manager.broadcast)

        try:
            yield
        finally:
            await streaming_service.shutdown()
            await alert_manager.shutdown()
            with suppress(Exception):
                await mqtt_service.stop()
            if not redis_provided:
                await redis.close()
                with suppress(AttributeError):
                    await redis.wait_closed()
            if not transactor_provided:
                with suppress(Exception):
                    await solana_transactor.close()

    app = FastAPI(
        title="DETECH Backend API",
        description="Real-time ingestion node for DETECH",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "DETECH Backend",
            "version": app.version,
            "network": settings.solana_network,
        }

    app.include_router(stream_router)

    return app


app = create_app()

