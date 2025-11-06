'use client';

import { useCallback, useEffect, useRef } from 'react';
import toast, { Toaster } from 'react-hot-toast';

import type { Alert } from '@/types';

type AlertSeverity = Alert['severity'];

const severityBadgeColors: Record<AlertSeverity, string> = {
  critical: 'bg-red-500/20 text-red-200 border border-red-500/40',
  high: 'bg-orange-500/20 text-orange-200 border border-orange-500/40',
  medium: 'bg-yellow-500/20 text-yellow-100 border border-yellow-500/30',
  low: 'bg-cyan-500/20 text-cyan-100 border border-cyan-500/30',
};

interface AlertsPanelProps {
  alerts: Alert[];
  onSendAlexa?: (alert: Alert) => Promise<void> | void;
  className?: string;
}

const paymentStatusCopy = (alert: Alert) => {
  if (alert.paymentStatus === 'confirmed') {
    return 'Charged 0.05 USDC';
  }
  if (alert.paymentStatus === 'pending') {
    return 'Payment pending';
  }
  if (alert.paymentStatus === 'failed') {
    return 'Payment failed';
  }
  return null;
};

const formatTimestamp = (timestamp: number) =>
  new Date(timestamp).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

export function AlertsPanel({ alerts, onSendAlexa, className }: AlertsPanelProps) {
  const latestToastId = useRef<string | null>(null);

  const handleSendAlexa = useCallback(
    async (alert: Alert) => {
      try {
        await onSendAlexa?.(alert);
        toast.success('queued command for Alexa bridge');
      } catch (error) {
        console.error('Alexa mock failed', error);
        toast.error('Alexa relay unavailable');
      }
    },
    [onSendAlexa]
  );

  useEffect(() => {
    if (!alerts.length) {
      return;
    }
    const topAlert = alerts[0];
    if (topAlert.id === latestToastId.current) {
      return;
    }
    latestToastId.current = topAlert.id;

    const commandsCopy = Array.isArray(topAlert.commands) && topAlert.commands.length > 0
      ? topAlert.commands.join(', ')
      : null;

    toast.custom(
      (t) => (
        <div
          className={`pointer-events-auto w-80 rounded-2xl border border-white/10 bg-slate-900/95 p-4 shadow-xl shadow-black/30 transition-all ${
            t.visible ? 'animate-in fade-in zoom-in' : 'animate-out fade-out zoom-out'
          }`}
          data-testid="swarm-alert-toast"
        >
          <div className="flex items-center justify-between text-xs uppercase tracking-wide text-white/60">
            <span>Swarm Alert</span>
            <span>{formatTimestamp(topAlert.timestamp)}</span>
          </div>
          <div className="mt-2 flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-semibold text-white">{topAlert.type}</p>
              <p className="mt-1 text-sm text-white/70">{topAlert.description}</p>
            </div>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${severityBadgeColors[topAlert.severity]}`}
            >
              {topAlert.severity}
            </span>
          </div>
          {commandsCopy && (
            <p className="mt-2 text-xs text-emerald-300/80">Gesture swarm commands: {commandsCopy}</p>
          )}
          {paymentStatusCopy(topAlert) && (
            <p className="mt-2 text-xs text-emerald-200/90">{paymentStatusCopy(topAlert)}</p>
          )}
          <div className="mt-3 flex gap-2 text-xs text-slate-300/80">
            <button
              onClick={async () => {
                await handleSendAlexa(topAlert);
                toast.dismiss(t.id);
              }}
              className="rounded-full bg-emerald-500/20 px-3 py-1 font-semibold text-emerald-200 transition hover:bg-emerald-400/30 hover:text-emerald-50"
            >
              Send to Alexa
            </button>
            <button
              onClick={() => toast.dismiss(t.id)}
              className="rounded-full bg-white/10 px-3 py-1 font-medium text-white/60 transition hover:bg-white/20 hover:text-white"
            >
              Dismiss
            </button>
          </div>
        </div>
      ),
      { id: topAlert.id, duration: 6000 }
    );
  }, [alerts, handleSendAlexa]);

  return (
    <div className={className}>
      <Toaster position="top-right" gutter={12} />
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Swarm Alerts</h3>
        <span className="text-xs text-white/40">Gesture mode routes commands live</span>
      </div>
      <div className="mt-4 space-y-3">
        {alerts.length === 0 ? (
          <div className="rounded-xl border border-white/5 bg-white/5 p-4 text-sm text-white/50">
            Awaiting detections. Gesture-driven swarm actions will appear here.
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className="rounded-xl border border-white/10 bg-slate-950/60 p-4 text-sm text-white/80 backdrop-blur"
              data-testid="alert-card"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-xs text-white/50">
                    <span>{formatTimestamp(alert.timestamp)}</span>
                    <span>•</span>
                    <span className={`rounded-full px-2 py-0.5 font-semibold ${severityBadgeColors[alert.severity]}`}>
                      {alert.severity}
                    </span>
                  </div>
                  <p className="mt-2 text-sm font-semibold text-white">{alert.type}</p>
                  <p className="mt-1 text-sm text-white/70">{alert.description}</p>
                  {alert.commands && alert.commands.length > 0 && (
                    <p className="mt-2 text-xs text-emerald-300/80">
                      Commands: {alert.commands.join(', ')}
                    </p>
                  )}
                  {alert.detection?.gestures?.length ? (
                    <p className="mt-1 text-xs text-sky-300/80">
                      Gestures: {alert.detection.gestures.map((gesture) => gesture.name).join(', ')}
                    </p>
                  ) : null}
                  {paymentStatusCopy(alert) && (
                    <p className="mt-2 text-xs text-emerald-200/90">{paymentStatusCopy(alert)}</p>
                  )}
                </div>
                <button
                  onClick={() => handleSendAlexa(alert)}
                  className="whitespace-nowrap rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 hover:text-emerald-50"
                >
                  Send to Alexa
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default AlertsPanel;

