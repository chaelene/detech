'use client';

import { useEffect, useMemo, useRef } from 'react';
import { Group, Layer, Rect, Stage, Text } from 'react-konva';
import type Konva from 'konva';

import type { BoundingBox } from '@/types';

export interface ActiveDetection {
  id: string;
  label: string;
  confidence: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  bbox: BoundingBox;
  createdAt: number;
  sourceAlertId: string;
}

export interface AlertsOverlayProps {
  detections: ActiveDetection[];
  containerSize: { width: number; height: number } | null;
  sourceSize: { width: number; height: number } | null;
}

const STROKE_COLORS: Record<ActiveDetection['severity'], string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#facc15',
  low: '#22d3ee',
};

const AnimatedRect = ({
  detection,
  rectProps,
}: {
  detection: ActiveDetection;
  rectProps: Konva.RectConfig;
}) => {
  const rectRef = useRef<Konva.Rect>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const rect = rectRef.current;
    if (!rect) {
      return undefined;
    }
    const layer = rect.getLayer();
    if (!layer) {
      return undefined;
    }
    const isPerson = detection.label.toLowerCase().includes('person');
    if (!isPerson) {
      rect.strokeWidth(2.5);
      rect.opacity(0.85);
      return undefined;
    }

    let startTime: number | null = null;

    const animate = (timestamp: number) => {
      if (startTime === null) {
        startTime = timestamp;
      }
      const elapsed = timestamp - startTime;
      const cycle = 1200;
      const t = (elapsed % cycle) / cycle;
      const pulse = 2 + Math.sin(t * Math.PI * 2) * 1.4;
      const opacity = 0.7 + Math.sin(t * Math.PI * 2) * 0.15;
      rect.strokeWidth(pulse);
      rect.opacity(opacity);
      rafRef.current = window.requestAnimationFrame(animate);
    };

    rafRef.current = window.requestAnimationFrame(animate);

    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      rect.strokeWidth(2.5);
      rect.opacity(0.85);
    };
  }, [detection.label]);

  return <Rect ref={rectRef} listening={false} {...rectProps} />;
};

export function AlertsOverlay({ detections, containerSize, sourceSize }: AlertsOverlayProps) {
  const now = Date.now();
  const activeDetections = useMemo(
    () => detections.filter((detection) => now - detection.createdAt < 4500),
    [detections, now]
  );

  const stageSize = useMemo(() => {
    if (!containerSize) {
      return null;
    }
    const source = sourceSize ?? containerSize;
    return {
      width: containerSize.width,
      height: containerSize.height,
      scaleX: containerSize.width / Math.max(1, source.width),
      scaleY: containerSize.height / Math.max(1, source.height),
    };
  }, [containerSize, sourceSize]);

  if (!stageSize || activeDetections.length === 0) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0" data-testid="alerts-overlay">
      <Stage width={stageSize.width} height={stageSize.height} listening={false} className="w-full h-full">
        <Layer>
          {activeDetections.map((detection) => {
            const { bbox } = detection;
            const x = bbox.x * stageSize.scaleX;
            const y = bbox.y * stageSize.scaleY;
            const width = bbox.width * stageSize.scaleX;
            const height = bbox.height * stageSize.scaleY;
            const stroke = STROKE_COLORS[detection.severity] ?? '#22d3ee';

            return (
              <Group key={detection.id} listening={false}>
                <AnimatedRect
                  detection={detection}
                  rectProps={{
                    x,
                    y,
                    width,
                    height,
                    stroke,
                    strokeWidth: 2.5,
                    dash: [6, 4],
                    dashEnabled: true,
                    shadowColor: stroke,
                    shadowBlur: 16,
                    cornerRadius: 6,
                  }}
                />
                <Group x={x} y={Math.max(4, y - 28)}>
                  <Rect listening={false} width={Math.max(120, width * 0.6)} height={24} fill="rgba(0,0,0,0.68)" cornerRadius={6} />
                  <Text
                    listening={false}
                    x={8}
                    y={5}
                    text={`${detection.label} • ${Math.round(detection.confidence * 100)}%`}
                    fontSize={12}
                    fill="#f8fafc"
                    fontFamily="Inter, system-ui, sans-serif"
                  />
                </Group>
              </Group>
            );
          })}
        </Layer>
      </Stage>
    </div>
  );
}

export default AlertsOverlay;

