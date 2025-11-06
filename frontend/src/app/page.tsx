'use client';

import { useCallback, useState } from 'react';

import Stream from '@/components/Stream';

export default function Home() {
  const [gestureMode, setGestureMode] = useState(false);

  const handleGestureModeChange = useCallback((enabled: boolean) => {
    setGestureMode(enabled);
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 sm:px-6 lg:px-8">
        <header className="space-y-3">
          <span className="inline-flex w-fit items-center gap-2 rounded-full bg-emerald-500/10 px-4 py-1 text-xs font-semibold uppercase tracking-widest text-emerald-300">
            Browser Dashcam • Swarm Edge Network
          </span>
          <h1 className="text-3xl font-bold leading-tight sm:text-4xl">DETECH Live Ops Console</h1>
          <p className="max-w-2xl text-sm text-white/70 sm:text-base">
            Capture an HTTPS-secured WebRTC feed from any mobile browser, negotiate it with Mediasoup,
            and watch swarm-refined anomaly alerts land in real time.
          </p>
        </header>

        <Stream gestureMode={gestureMode} onGestureModeChange={handleGestureModeChange} />
      </div>
    </main>
  );
}
