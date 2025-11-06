"""
Solana x402 micro-payment helpers
"""

from solana.rpc.api import Client
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from typing import Optional
import base58


class X402Payment:
    """Helper class for x402 micro-payments on Solana"""
    
    def __init__(self, rpc_url: str, program_id: Optional[str] = None):
        """
        Initialize x402 payment helper
        
        Args:
            rpc_url: Solana RPC endpoint URL
            program_id: x402 program ID (optional, will use default if not provided)
        """
        self.client = Client(rpc_url)
        self.program_id = program_id or "x402_program_id_placeholder"  # TODO: Get actual program ID
    
    def create_payment_transaction(
        self,
        sender: Keypair,
        recipient: str,
        amount_usdc: float,
        alert_id: str,
    ) -> Transaction:
        """
        Create x402 payment transaction
        
        Args:
            sender: Sender keypair
            recipient: Recipient Solana address
            amount_usdc: Amount in USDC (will be converted to smallest unit)
            alert_id: Alert ID for payment tracking
            
        Returns:
            Transaction object ready to be signed and sent
        """
        # TODO: Implement actual x402 transaction creation
        # This would:
        # 1. Create a transaction invoking the x402 program
        # 2. Include alert_id as metadata/instruction data
        # 3. Set USDC amount and recipient
        # 4. Return unsigned transaction
        
        recipient_pubkey = Pubkey.from_string(recipient)
        
        # Convert USDC to smallest unit (6 decimals)
        amount_smallest_unit = int(amount_usdc * 1_000_000)
        
        # Placeholder transaction
        # In production, this would create a proper x402 program instruction
        transaction = Transaction()
        
        # TODO: Add x402 program instruction
        
        return transaction
    
    def send_payment(
        self,
        transaction: Transaction,
        sender: Keypair,
    ) -> Optional[str]:
        """
        Send payment transaction to Solana network
        
        Args:
            transaction: Transaction to send
            sender: Sender keypair for signing
            
        Returns:
            Transaction signature if successful, None otherwise
        """
        try:
            # Sign transaction
            transaction.sign(sender)
            
            # Send transaction
            response = self.client.send_transaction(transaction, sender)
            
            if response.value:
                return str(response.value)
            else:
                return None
                
        except Exception as e:
            print(f"Error sending payment: {e}")
            return None
    
    def verify_payment(self, tx_signature: str) -> bool:
        """
        Verify that a payment transaction was confirmed
        
        Args:
            tx_signature: Transaction signature to verify
            
        Returns:
            True if transaction is confirmed, False otherwise
        """
        try:
            response = self.client.get_transaction(tx_signature)
            return response.value is not None
        except Exception as e:
            print(f"Error verifying payment: {e}")
            return False
    
    def get_payment_amount(self, tx_signature: str) -> Optional[float]:
        """
        Get payment amount from transaction
        
        Args:
            tx_signature: Transaction signature
            
        Returns:
            Payment amount in USDC, or None if not found
        """
        try:
            response = self.client.get_transaction(tx_signature)
            if response.value:
                # TODO: Parse transaction to extract USDC amount
                # This would require parsing the x402 program instruction
                return None
            return None
        except Exception as e:
            print(f"Error getting payment amount: {e}")
            return None
