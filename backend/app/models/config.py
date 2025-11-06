"""
Configuration models for DETECH Backend
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]
    
    # Mediasoup WebRTC configuration
    mediasoup_worker_path: str = "/usr/local/bin/mediasoup-worker"
    mediasoup_router_rtp_ips: List[str] = ["127.0.0.1"]
    frame_sample_interval_seconds: float = 5.0
    session_ttl_seconds: int = 3600
    rate_limit_window_seconds: int = 60
    rate_limit_max_streams: int = 20
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # MQTT configuration
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic_frames: str = "detech/frames"
    mqtt_topic_detections: str = "detech/detections"
    mqtt_topic_alerts: str = "detech/alerts"
    mqtt_reconnect_interval_seconds: float = 2.0
    
    # Solana configuration
    solana_rpc_url: str = "https://rpc.helius.xyz"
    solana_network: str = "mainnet-beta"
    solana_wallet_private_key: str = ""  # TODO: Load from secure storage
    x402_program_id: str = ""  # TODO: x402 program ID
    x402_charge_per_interval_usdc: float = 0.05
    x402_charge_interval_seconds: int = 60
    x402_agent_private_key: str = ""
    x402_usdc_mint: str = "EPjFWdd5AufqSSqeM2qoi9GAJh9zn7wo7GSaJ6y4wDR"
    x402_max_retries: int = 3
    x402_retry_backoff_seconds: float = 0.5
    xai_api_key: str = ""
    openai_api_key: str = ""
    x402_mock_transfers: bool = True
    
    # Swarm configuration
    swarm_api_url: str = "http://localhost:8001"
    swarm_mock_mode: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
