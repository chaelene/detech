"""
Solana x402 payment service
"""

from solana.rpc.api import Client
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.rpc.responses import SendTransactionResp
from typing import Optional
import base58
import os
from app.models.config import Settings

settings = Settings()


class SolanaService:
    """Service for handling Solana x402 micro-payments"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.wallet: Optional[Keypair] = None
        self._ready = False
    
    async def initialize(self):
        """Initialize Solana client and wallet"""
        try:
            self.client = Client(settings.solana_rpc_url)
            
            # Load wallet from private key
            if settings.solana_wallet_private_key:
                private_key_bytes = base58.b58decode(settings.solana_wallet_private_key)
                self.wallet = Keypair.from_secret_key(private_key_bytes)
            else:
                # TODO: Generate or load from secure storage
                print("Warning: No wallet private key configured")
            
            self._ready = True
            print("SolanaService initialized")
        except Exception as e:
            print(f"Failed to initialize SolanaService: {e}")
            self._ready = False
    
    async def send_x402_payment(
        self,
        recipient: str,
        amount_usdc: float,
        alert_id: str
    ) -> Optional[str]:
        """
        Send x402 micro-payment in USDC
        
        Args:
            recipient: Solana address of recipient
            amount_usdc: Amount in USDC (will be converted to smallest unit)
            alert_id: Alert ID for tracking
            
        Returns:
            Transaction signature if successful, None otherwise
        """
        if not self._ready or not self.wallet:
            print("SolanaService not ready")
            return None
        
        try:
            recipient_pubkey = Pubkey.from_string(recipient)
            
            # TODO: Implement x402 protocol payment
            # x402 is a micro-payment protocol on Solana
            # For now, we'll create a basic transfer structure
            
            # Convert USDC amount to smallest unit (6 decimals)
            amount_lamports = int(amount_usdc * 1_000_000)
            
            # TODO: Create x402 payment transaction
            # This should interact with the x402 program on Solana
            # For now, we'll create a placeholder structure
            
            # In production, this would:
            # 1. Create a transaction invoking the x402 program
            # 2. Include the alert_id as metadata
            # 3. Send USDC to recipient
            # 4. Return transaction signature
            
            print(f"Sending x402 payment: {amount_usdc} USDC to {recipient} for alert {alert_id}")
            
            # Placeholder - actual implementation needed
            # transaction = create_x402_transaction(...)
            # signature = await self.client.send_transaction(transaction, self.wallet)
            
            return "placeholder_tx_signature"
            
        except Exception as e:
            print(f"Failed to send x402 payment: {e}")
            return None
    
    async def verify_payment(self, tx_signature: str) -> bool:
        """Verify that a payment transaction was confirmed"""
        if not self.client:
            return False
        
        try:
            # Check transaction status
            response = self.client.get_transaction(tx_signature)
            return response.value is not None
        except Exception as e:
            print(f"Failed to verify payment: {e}")
            return False
    
    def get_wallet_address(self) -> Optional[str]:
        """Get the wallet public address"""
        if self.wallet:
            return str(self.wallet.pubkey())
        return None
    
    async def cleanup(self):
        """Cleanup resources"""
        self._ready = False
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready
