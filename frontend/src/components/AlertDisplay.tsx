'use client';

import { Alert } from '@/types';

interface AlertDisplayProps {
  alerts: Alert[];
}

const severityPalette: Record<Alert['severity'], { border: string; background: string }> = {
  critical: { border: 'border-red-500', background: 'bg-red-50' },
  high: { border: 'border-orange-500', background: 'bg-orange-50' },
  medium: { border: 'border-yellow-500', background: 'bg-yellow-50' },
  low: { border: 'border-blue-500', background: 'bg-blue-50' },
};

export function AlertDisplay({ alerts }: AlertDisplayProps) {
  if (alerts.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        No alerts. Stream will appear here when detections are made.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className={`p-4 rounded-lg border-2 ${severityPalette[alert.severity].border} ${severityPalette[alert.severity].background}`}
        >
          <div className="flex justify-between items-start mb-2">
            <h3 className="font-semibold">{alert.type}</h3>
            <span className="text-xs text-gray-500">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </span>
          </div>
          <p className="text-sm mb-2">{alert.description}</p>
          {alert.detection && (
            <div className="text-xs text-gray-600 mb-2 space-y-1">
              {alert.detection.objects.map((object, index) => (
                <div key={`${alert.id}-object-${index}`}>
                  <span className="font-semibold">Object:</span> {object.name}
                  <span className="ml-1">({Math.round(object.confidence * 100)}% confidence)</span>
                </div>
              ))}
              {alert.detection.gestures.map((gesture, index) => (
                <div key={`${alert.id}-gesture-${index}`}>
                  <span className="font-semibold">Gesture:</span> {gesture.name}
                  <span className="ml-1">({Math.round(gesture.confidence * 100)}% confidence)</span>
                </div>
              ))}
            </div>
          )}
          {alert.paymentStatus && (
            <div className="text-xs">
              Payment: {alert.paymentStatus}
              {alert.paymentTxId && (
                <a
                  href={`https://solscan.io/tx/${alert.paymentTxId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 text-blue-600 hover:underline"
                >
                  View TX
                </a>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
