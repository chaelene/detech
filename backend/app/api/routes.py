"""
API routes for DETECH Backend
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.services.solana import SolanaService
from app.services.streaming import StreamingService

router = APIRouter()


class PaymentRequest(BaseModel):
    recipient: str
    amount_usdc: float
    alert_id: str


class PaymentResponse(BaseModel):
    tx_signature: str
    status: str


@router.get("/streams")
async def list_streams():
    """List active streams"""
    # TODO: Get from streaming service
    return {"streams": []}


@router.post("/payment", response_model=PaymentResponse)
async def create_payment(
    request: PaymentRequest,
    solana_service: SolanaService = Depends(lambda: SolanaService())
):
    """Create x402 payment for alert"""
    tx_signature = await solana_service.send_x402_payment(
        recipient=request.recipient,
        amount_usdc=request.amount_usdc,
        alert_id=request.alert_id
    )
    
    if not tx_signature:
        raise HTTPException(status_code=500, detail="Failed to create payment")
    
    return PaymentResponse(
        tx_signature=tx_signature,
        status="pending"
    )


@router.get("/payment/{tx_signature}")
async def verify_payment(
    tx_signature: str,
    solana_service: SolanaService = Depends(lambda: SolanaService())
):
    """Verify payment transaction"""
    verified = await solana_service.verify_payment(tx_signature)
    return {
        "tx_signature": tx_signature,
        "verified": verified
    }


@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
