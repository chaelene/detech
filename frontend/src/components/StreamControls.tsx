'use client';

interface StreamControlsProps {
  isStreaming: boolean;
  onStart: () => void;
  onStop: () => void;
  onConnectWallet: () => void;
  onDisconnectWallet: () => void;
  walletAddress: string | null;
  isWalletConnected: boolean;
}

export function StreamControls({
  isStreaming,
  onStart,
  onStop,
  onConnectWallet,
  onDisconnectWallet,
  walletAddress,
  isWalletConnected,
}: StreamControlsProps) {
  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <button
          onClick={isStreaming ? onStop : onStart}
          className={`px-6 py-2 rounded-lg font-semibold ${
            isStreaming
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isStreaming ? 'Stop Stream' : 'Start Stream'}
        </button>

        {isWalletConnected ? (
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">
              {walletAddress?.slice(0, 4)}...{walletAddress?.slice(-4)}
            </span>
            <button
              onClick={onDisconnectWallet}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg text-sm"
            >
              Disconnect
            </button>
          </div>
        ) : (
          <button
            onClick={onConnectWallet}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm"
          >
            Connect Wallet
          </button>
        )}
      </div>
    </div>
  );
}
