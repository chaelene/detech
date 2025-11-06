import { useState, useCallback, useEffect, useMemo } from 'react';
import { Connection, PublicKey } from '@solana/web3.js';

const SOLANA_RPC_URL = process.env.NEXT_PUBLIC_SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';

type PhantomConnectOpts = {
  onlyIfTrusted?: boolean;
};

type PhantomProvider = {
  isPhantom?: boolean;
  publicKey?: { toString: () => string };
  connect: (options?: PhantomConnectOpts) => Promise<{ publicKey: { toString: () => string } }>;
  disconnect: () => Promise<void>;
  on: (event: string, handler: (...args: unknown[]) => void) => void;
  request: (args: { method: string; params?: unknown[] }) => Promise<unknown>;
};

type ListenerCapableProvider = PhantomProvider & {
  removeListener?: (event: string, handler: (...args: unknown[]) => void) => void;
  off?: (event: string, handler: (...args: unknown[]) => void) => void;
};

function resolvePhantomProvider(): PhantomProvider | null {
  if (typeof window === 'undefined') {
    return null;
  }
  const solana = (window as any).solana ?? (window as any).phantom?.solana;
  if (solana && solana.isPhantom) {
    return solana as PhantomProvider;
  }
  return null;
}

export function useWallet() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [provider, setProvider] = useState<PhantomProvider | null>(() => resolvePhantomProvider());
  const [connection] = useState(() => new Connection(SOLANA_RPC_URL, 'confirmed'));

  useEffect(() => {
    setProvider(resolvePhantomProvider());
  }, []);

  useEffect(() => {
    if (!provider || typeof provider.on !== 'function') {
      return undefined;
    }

    const handleConnect = (publicKey: { toString: () => string }) => {
      const pubKeyString = publicKey.toString();
      setWalletAddress(pubKeyString);
      setIsConnected(true);
    };

    const handleDisconnect = () => {
      setWalletAddress(null);
      setIsConnected(false);
    };

    const handleAccountChanged = (newPk: { toString: () => string } | null) => {
      if (newPk) {
        handleConnect(newPk);
      } else {
        handleDisconnect();
      }
    };

    const connectListener = (...args: unknown[]) => {
      const [pk] = args;
      if (pk && typeof (pk as { toString?: unknown }).toString === 'function') {
        handleConnect(pk as { toString: () => string });
      }
    };

    const disconnectListener = () => {
      handleDisconnect();
    };

    const accountChangedListener = (...args: unknown[]) => {
      const [pk] = args;
      if (pk && typeof (pk as { toString?: unknown }).toString === 'function') {
        handleAccountChanged(pk as { toString: () => string });
      } else {
        handleAccountChanged(null);
      }
    };

    provider.on('connect', connectListener);
    provider.on('disconnect', disconnectListener);
    provider.on('accountChanged', accountChangedListener);

    return () => {
      const listenerCapable = provider as ListenerCapableProvider;
      const maybeRemove = (event: string, handler: (...args: unknown[]) => void) => {
        if (typeof listenerCapable.off === 'function') {
          listenerCapable.off(event, handler);
        } else if (typeof listenerCapable.removeListener === 'function') {
          listenerCapable.removeListener(event, handler);
        }
      };

      maybeRemove('connect', connectListener);
      maybeRemove('disconnect', disconnectListener);
      maybeRemove('accountChanged', accountChangedListener);
    };
  }, [provider]);

  const connectWallet = useCallback(async () => {
    if (!provider) {
      throw new Error('Phantom wallet not found');
    }
    try {
      const response = await provider.connect({ onlyIfTrusted: false });
      const pubKey = response.publicKey.toString();
      setWalletAddress(pubKey);
      setIsConnected(true);
      return pubKey;
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      throw error;
    }
  }, [provider]);

  const disconnectWallet = useCallback(async () => {
    if (!provider) {
      return;
    }
    try {
      await provider.disconnect();
    } catch (error) {
      console.error('Failed to disconnect wallet:', error);
      throw error;
    } finally {
      setWalletAddress(null);
      setIsConnected(false);
    }
  }, [provider]);

  const sendPayment = useCallback(
    async (amount: number, recipient: string) => {
      if (!isConnected || !walletAddress) {
        throw new Error('Wallet not connected');
      }

      try {
        const recipientPubkey = new PublicKey(recipient);
        console.log(`Simulating x402 payment of ${amount} USDC to ${recipientPubkey.toString()}`);
        // TODO: Integrate x402 payment flow with backend signer
      } catch (error) {
        console.error('Failed to send payment:', error);
        throw error;
      }
    },
    [isConnected, walletAddress]
  );

  const installUrl = useMemo(() => 'https://phantom.app/download', []);

  return {
    connectWallet,
    disconnectWallet,
    sendPayment,
    walletAddress,
    isConnected,
    connection,
    provider,
    isPhantomAvailable: Boolean(provider),
    installUrl,
  };
}
