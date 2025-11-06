'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

import { Alert, AlertSocketPayload, StreamAnswerPayload, StreamOfferPayload } from '@/types';
import { useWallet } from '@/lib/wallet';
import AlertsPanel from '@/components/Alerts';
import type { ActiveDetection, AlertsOverlayProps } from '@/components/AlertsOverlay';
import { createWsClient, type WsClient } from '@/utils/ws_client';

const AlertsOverlay = dynamic<AlertsOverlayProps>(
  () => import('@/components/AlertsOverlay').then((mod) => mod.AlertsOverlay),
  { ssr: false }
);

interface StreamProps {
  gestureMode: boolean;
  onGestureModeChange?: (enabled: boolean) => void;
}

const BOX_TTL_MS = 4_500;
const ALERT_LIMIT = 30;

const DEFAULT_SEVERITY: Alert['severity'] = 'medium';

const ICE_SERVER_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }],
};

function normaliseSeverity(value: unknown): Alert['severity'] {
  if (value === 'low' || value === 'medium' || value === 'high' || value === 'critical') {
    return value;
  }
  return DEFAULT_SEVERITY;
}

function clampBoundingBox(box: Partial<Record<'x' | 'y' | 'width' | 'height', number>>): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  return {
    x: Math.max(0, box.x ?? 0),
    y: Math.max(0, box.y ?? 0),
    width: Math.max(0, box.width ?? 0),
    height: Math.max(0, box.height ?? 0),
  };
}

function decodeBase64(value: string): string {
  if (typeof window !== 'undefined' && typeof window.atob === 'function') {
    return window.atob(value);
  }
  if (typeof globalThis !== 'undefined' && typeof (globalThis as { atob?: (data: string) => string }).atob === 'function') {
    return (globalThis as { atob: (data: string) => string }).atob(value);
  }
  throw new Error('Base64 decoding is not supported in this environment');
}

function encodeBase64(value: string): string {
  if (typeof window !== 'undefined' && typeof window.btoa === 'function') {
    return window.btoa(value);
  }
  if (typeof globalThis !== 'undefined' && typeof (globalThis as { btoa?: (data: string) => string }).btoa === 'function') {
    return (globalThis as { btoa: (data: string) => string }).btoa(value);
  }
  throw new Error('Base64 encoding is not supported in this environment');
}

export function Stream({ gestureMode, onGestureModeChange }: StreamProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const wsRef = useRef<WsClient<AlertSocketPayload> | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [activeDetections, setActiveDetections] = useState<ActiveDetection[]>([]);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number } | null>(null);
  const [containerElement, setContainerElement] = useState<HTMLDivElement | null>(null);
  const [sessionId] = useState<string>(() => {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return `session-${Date.now().toString(36)}`;
  });
  const [isStreaming, setIsStreaming] = useState(false);
  const [isNegotiating, setIsNegotiating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('Camera idle');
  const [videoResolution, setVideoResolution] = useState<{ width: number; height: number } | null>(null);
  const [sourceResolution, setSourceResolution] = useState<{ width: number; height: number } | null>(null);
  const assignVideoContainerRef = useCallback((node: HTMLDivElement | null) => {
    videoContainerRef.current = node;
    setContainerElement(node);
    if (node) {
      const rect = node.getBoundingClientRect();
      setContainerSize({ width: rect.width, height: rect.height });
    }
  }, []);
  const { connectWallet, disconnectWallet, walletAddress, isConnected, isPhantomAvailable, installUrl } = useWallet();

  const backendBaseUrl = useMemo(() => {
    const url = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
    return url.replace(/\/$/, '');
  }, []);

  const enforceHttps = useCallback(() => {
    if (typeof window === 'undefined') {
      return true;
    }
    if (window.isSecureContext) {
      return true;
    }
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return true;
    }
    setErrorMessage('Camera access is only available over HTTPS. Please reload the page using https://');
    return false;
  }, []);

  const waitForIceGatheringComplete = useCallback((pc: RTCPeerConnection) => {
    if (pc.iceGatheringState === 'complete') {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', checkState);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', checkState);
    });
  }, []);

  const cleanupMedia = useCallback(() => {
    peerRef.current?.close();
    peerRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setActiveDetections([]);
  }, []);

  useEffect(() => {
    return () => {
      cleanupMedia();
    };
  }, [cleanupMedia]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      const cutoff = Date.now() - BOX_TTL_MS;
      setActiveDetections((current) => current.filter((box) => box.createdAt >= cutoff));
    }, 1_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || !containerElement) {
      return;
    }
    const element = containerElement;

    const updateSize = () => {
      const rect = element.getBoundingClientRect();
      setContainerSize({ width: rect.width, height: rect.height });
    };

    updateSize();

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateSize);
      return () => {
        window.removeEventListener('resize', updateSize);
      };
    }

    const observer = new ResizeObserver((entries) => {
      entries.forEach((entry) => {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      });
    });
    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [containerElement]);

  const normaliseAlertPayload = useCallback((payload: AlertSocketPayload): Alert => {
    const detectionPayload = payload.detection ?? (payload as any).detection ?? null;
    const objects = Array.isArray(detectionPayload?.objects)
      ? detectionPayload.objects.map((object: any) => ({
          name: String(object?.name ?? 'unknown'),
          confidence: Number(object?.confidence ?? 0),
          bbox: clampBoundingBox({
            x: object?.bbox?.[0],
            y: object?.bbox?.[1],
            width: object?.bbox?.[2],
            height: object?.bbox?.[3],
          }),
        }))
      : [];
    const gestures = Array.isArray(detectionPayload?.gestures)
      ? detectionPayload.gestures.map((gesture: any) => ({
          name: String(gesture?.name ?? 'gesture'),
          confidence: Number(gesture?.confidence ?? 0),
          landmarks: Array.isArray(gesture?.landmarks) ? gesture.landmarks : undefined,
        }))
      : [];

    const detection = detectionPayload
      ? {
          timestamp: Number(detectionPayload.timestamp ?? Date.now()),
          objects,
          gestures,
        }
      : undefined;

    const severity = normaliseSeverity(payload.severity);

    const description =
      (typeof payload.description === 'string' && payload.description.length > 0
        ? payload.description
        : undefined) ??
      (typeof payload.label === 'string' ? payload.label : '') ??
      'Alert received from swarm';

    return {
      id: String(payload.alert_id ?? payload.id ?? `${payload.session_id}-${payload.timestamp ?? Date.now()}`),
      sessionId: payload.session_id,
      timestamp: Number(payload.timestamp ?? Date.now()),
      type: typeof payload.type === 'string' && payload.type.length > 0 ? payload.type : 'event',
      severity,
      description,
      detection,
      source: (payload.source === 'jetson' || payload.source === 'swarm') ? payload.source : 'swarm',
      paymentStatus: payload.payment_status as Alert['paymentStatus'],
      paymentTxId: payload.payment_tx_id,
      paymentAmount: payload.payment_amount,
      swarmConfidence: payload.swarm_confidence,
      commands: Array.isArray(payload.commands) ? (payload.commands as string[]) : [],
      threatLevel: payload.threat_level,
      anomalyScore: payload.anomaly_score,
      x402: (payload as Record<string, unknown>).x402 as Record<string, unknown> | undefined,
      metadata: payload.metadata,
    };
  }, []);

  const handleAlertMessage = useCallback(
    (payload: AlertSocketPayload) => {
      const alert = normaliseAlertPayload(payload);
      setAlerts((current) => [alert, ...current].slice(0, ALERT_LIMIT));
      if (alert.detection?.objects.length) {
        const createdAt = Date.now();
        setActiveDetections((current) => {
          const next = current.filter((box) => createdAt - box.createdAt < BOX_TTL_MS);
          alert.detection?.objects.forEach((object, index) => {
            next.push({
              id: `${alert.id}-object-${index}`,
              label: object.name,
              confidence: object.confidence,
              severity: alert.severity,
              bbox: {
                x: object.bbox.x,
                y: object.bbox.y,
                width: object.bbox.width,
                height: object.bbox.height,
              },
              createdAt,
              sourceAlertId: alert.id,
            });
          });
          return next;
        });
      }
    },
    [normaliseAlertPayload]
  );

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }
    const handleMockAlert = (event: Event) => {
      const customEvent = event as CustomEvent<AlertSocketPayload | AlertSocketPayload[]>;
      const detail = customEvent.detail;
      if (!detail) {
        return;
      }
      const payloads = Array.isArray(detail) ? detail : [detail];
      payloads.forEach((payload) => handleAlertMessage(payload));
    };

    window.addEventListener('detech:mock-alert', handleMockAlert);

    return () => {
      window.removeEventListener('detech:mock-alert', handleMockAlert);
    };
  }, [handleAlertMessage]);

  useEffect(() => {
    const client = wsRef.current;
    if (!client) {
      return;
    }
    client.sendJson({
      type: 'gesture_mode_update',
      session_id: sessionId,
      gesture_mode: gestureMode,
      swarm_input: gestureMode ? 'gestures' : 'manual',
    });
  }, [gestureMode, sessionId]);

  const openAlertSocket = useCallback(
    (session: string) => {
      const wsUrl = `${backendBaseUrl.replace(/^http/, 'ws')}/alerts?session_id=${encodeURIComponent(session)}`;
      const client = createWsClient<AlertSocketPayload>({
        url: wsUrl,
        onOpen: () => {
          setStatusMessage('Connected to swarm alerts');
          if (typeof queueMicrotask === 'function') {
            queueMicrotask(() => {
              wsRef.current?.sendJson({
                type: 'gesture_mode_update',
                session_id: session,
                gesture_mode: gestureMode,
                swarm_input: gestureMode ? 'gestures' : 'manual',
              });
            });
          } else {
            setTimeout(() => {
              wsRef.current?.sendJson({
                type: 'gesture_mode_update',
                session_id: session,
                gesture_mode: gestureMode,
                swarm_input: gestureMode ? 'gestures' : 'manual',
              });
            }, 0);
          }
        },
        onMessage: (payload) => {
          handleAlertMessage(payload);
        },
        onError: () => {
          setStatusMessage('Alert channel error');
          setErrorMessage('Encountered an error receiving alerts.');
        },
        onClose: () => {
          setStatusMessage('Alert channel closed');
        },
      });
      wsRef.current = client;
    },
    [backendBaseUrl, gestureMode, handleAlertMessage]
  );

  const startStream = useCallback(async () => {
    if (isNegotiating || isStreaming) {
      return;
    }
    if (!enforceHttps()) {
      return;
    }
    setErrorMessage(null);
    setStatusMessage('Initialising camera...');
    setIsNegotiating(true);

    try {
      let activeWallet = walletAddress;
      if (!isConnected) {
        activeWallet = await connectWallet();
      }
      if (!activeWallet) {
        throw new Error('Wallet connection required');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 60 },
        },
      });
      mediaStreamRef.current = stream;
      const videoElement = videoRef.current;
      if (videoElement) {
        videoElement.srcObject = stream;
        await videoElement.play();
        setVideoResolution({ width: videoElement.videoWidth, height: videoElement.videoHeight });
      }

      const videoTrack = stream.getVideoTracks()[0];
      const settings = videoTrack?.getSettings();
      if (settings?.width && settings?.height) {
        setSourceResolution({ width: settings.width, height: settings.height });
      }

      const peer = new RTCPeerConnection(ICE_SERVER_CONFIG);
      peerRef.current = peer;
      const iceCandidates: RTCIceCandidateInit[] = [];

      peer.addEventListener('icecandidate', (event) => {
        if (event.candidate) {
          iceCandidates.push(event.candidate.toJSON());
        }
      });

      peer.addEventListener('connectionstatechange', () => {
        const state = peer.connectionState;
        if (state === 'connected') {
          setStatusMessage('Streaming to edge nodes');
          setIsStreaming(true);
        } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
          setStatusMessage('Stream disconnected');
          setIsStreaming(false);
          cleanupMedia();
        }
      });

      stream.getTracks().forEach((track) => peer.addTrack(track, stream));

      const offer = await peer.createOffer({ offerToReceiveAudio: false, offerToReceiveVideo: false });
      await peer.setLocalDescription(offer);
      await waitForIceGatheringComplete(peer);

      const localDescription = peer.localDescription;
      if (!localDescription) {
        throw new Error('Failed to create SDP offer');
      }

      const payload: StreamOfferPayload = {
        session_id: sessionId,
        wallet_pubkey: activeWallet,
        sdp: encodeBase64(localDescription.sdp),
        ice_candidates: iceCandidates,
        metadata: {
          user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
          constraints: settings,
          gesture_mode: gestureMode,
          swarm_input: gestureMode ? 'gestures' : 'manual',
          x402: {
            balance_usdc: null,
          },
        },
      };

      const response = await fetch(`${backendBaseUrl}/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const failureMessage = await response.text();
        throw new Error(failureMessage || 'Failed to negotiate stream');
      }

      const data = (await response.json()) as StreamAnswerPayload;
      const answerSdp = decodeBase64(data.answer_sdp);
      await peer.setRemoteDescription({ type: 'answer', sdp: answerSdp });
      setStatusMessage('Awaiting Mediasoup transport...');
      openAlertSocket(sessionId);
    } catch (error) {
      console.error('Failed to start stream', error);
      cleanupMedia();
      setIsStreaming(false);
      setErrorMessage(error instanceof Error ? error.message : 'Unexpected error starting stream');
      setStatusMessage('Idle');
    } finally {
      setIsNegotiating(false);
    }
  }, [
    backendBaseUrl,
    cleanupMedia,
    connectWallet,
    enforceHttps,
    gestureMode,
    isConnected,
    isNegotiating,
    isStreaming,
    openAlertSocket,
    sessionId,
    waitForIceGatheringComplete,
    walletAddress,
  ]);

  const stopStream = useCallback(() => {
    cleanupMedia();
    setIsStreaming(false);
    setStatusMessage('Stream stopped');
  }, [cleanupMedia]);

  const toggleGestureMode = useCallback(() => {
    onGestureModeChange?.(!gestureMode);
  }, [gestureMode, onGestureModeChange]);

  const handleSendAlexa = useCallback(async (alert: Alert) => {
    try {
      const response = await fetch('/api/mock/alexa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alert.id,
          commands: alert.commands ?? [],
          description: alert.description,
          severity: alert.severity,
        }),
      });
      if (!response.ok) {
        console.info('Alexa mock endpoint returned non-success status', response.status);
      }
    } catch (error) {
      console.info('Alexa mock POST failed, continuing optimistically', error);
    }
    await new Promise((resolve) => setTimeout(resolve, 320));
  }, []);

  const handleVideoLoadedMetadata = useCallback(() => {
    const element = videoRef.current;
    if (element && element.videoWidth && element.videoHeight) {
      setVideoResolution({ width: element.videoWidth, height: element.videoHeight });
    }
    if (videoContainerRef.current) {
      const rect = videoContainerRef.current.getBoundingClientRect();
      setContainerSize({ width: rect.width, height: rect.height });
    }
  }, []);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <section className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Browser Dashcam</h2>
            <p className="text-sm text-white/70">
              Phantom-secured WebRTC edge stream with swarm anomaly overlays.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {isConnected && walletAddress ? (
              <button
                onClick={disconnectWallet}
                className="rounded-full bg-white/10 px-4 py-2 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/20"
              >
                {walletAddress.slice(0, 4)}…{walletAddress.slice(-4)}
              </button>
            ) : isPhantomAvailable ? (
              <button
                onClick={connectWallet}
                className="rounded-full bg-violet-500 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-500/30 transition hover:bg-violet-600 active:scale-95"
              >
                Connect Phantom
              </button>
            ) : (
              <a
                href={installUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full bg-amber-500 px-5 py-2 text-sm font-semibold text-black shadow-lg shadow-amber-500/30 transition hover:bg-amber-400 active:scale-95"
              >
                Install Phantom
              </a>
            )}
            <button
              type="button"
              role="switch"
              aria-checked={gestureMode}
              onClick={toggleGestureMode}
              className={`flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-wide transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300 ${
                gestureMode
                  ? 'border-emerald-400/40 bg-emerald-500/20 text-emerald-200 shadow-inner shadow-emerald-400/30'
                  : 'border-white/20 bg-white/5 text-white/60 hover:bg-white/10'
              }`}
              data-testid="gesture-mode-toggle"
            >
              <span>Gesture Mode</span>
              <span className="relative inline-flex h-5 w-10 items-center rounded-full bg-black/60">
                <span
                  className={`inline-block h-4 w-4 rounded-full bg-emerald-400 transition-transform duration-200 ${
                    gestureMode ? 'translate-x-5 shadow shadow-emerald-400/40' : 'translate-x-1 opacity-60'
                  }`}
                />
              </span>
            </button>
            <button
              onClick={isStreaming ? stopStream : startStream}
              disabled={isNegotiating}
              className={`rounded-full px-6 py-2 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white active:scale-95 ${
                isStreaming
                  ? 'bg-red-500 text-white shadow-lg shadow-red-500/30 hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-60'
                  : 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30 hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-60'
              }`}
              data-testid="stream-toggle"
            >
              {isStreaming ? 'Stop Stream' : isNegotiating ? 'Connecting…' : 'Start Stream'}
            </button>
          </div>
        </div>

        <div
          ref={assignVideoContainerRef}
          className="relative aspect-video w-full overflow-hidden rounded-2xl border border-white/10 bg-black shadow-xl"
        >
          <video
            ref={videoRef}
            playsInline
            muted
            onLoadedMetadata={handleVideoLoadedMetadata}
            className="h-full w-full object-cover"
          />
          <AlertsOverlay
            detections={activeDetections}
            containerSize={containerSize}
            sourceSize={sourceResolution}
          />
          <div className="absolute left-4 top-4 inline-flex items-center gap-2 rounded-full bg-black/60 px-4 py-1 text-xs font-semibold text-white backdrop-blur">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                isStreaming ? 'bg-emerald-400 animate-pulse' : 'bg-yellow-300'
              }`}
            />
            <span>{statusMessage}</span>
            {gestureMode && (
              <span className="ml-2 rounded-full bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-200">
                Gesture
              </span>
            )}
          </div>
          {errorMessage && (
            <div className="absolute inset-x-4 bottom-4 rounded-xl bg-red-500/80 px-4 py-3 text-sm text-white shadow-lg">
              {errorMessage}
            </div>
          )}
        </div>

        {videoResolution && (
          <p className="text-xs text-white/60">
            Live feed: {Math.round(videoResolution.width)}×{Math.round(videoResolution.height)} | Session {sessionId}
          </p>
        )}
      </section>

      <AlertsPanel
        alerts={alerts}
        onSendAlexa={handleSendAlexa}
        className="min-h-[320px] rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur"
      />
    </div>
  );
}

export default Stream;

