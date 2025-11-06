import type { Meta, StoryObj } from '@storybook/react';
import { FC, ReactNode, useEffect } from 'react';

import Stream from './Stream';

const mockAlertPayload = {
  session_id: 'story-session',
  alert_id: 'alert-demo',
  type: 'object_detection',
  severity: 'high',
  description: 'Vehicle detected: potential obstruction ahead',
  timestamp: Date.now(),
  source: 'swarm',
  detection: {
    timestamp: Date.now(),
    objects: [
      {
        name: 'vehicle',
        confidence: 0.92,
        bbox: [140, 160, 260, 240],
      },
      {
        name: 'person',
        confidence: 0.81,
        bbox: [420, 180, 180, 360],
      },
    ],
    gestures: [],
  },
  commands: ['slow-down', 'prepare-stop'],
  swarm_confidence: 0.88,
};

const MockEnvironment: FC<{ children: ReactNode }> = ({ children }) => {
  useEffect(() => {
    const originalWebSocket = window.WebSocket;
    const originalSolana = (window as any).solana;
    const originalMediaDevices = navigator.mediaDevices?.getUserMedia;
    const originalBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

    type Handler = (...args: any[]) => void;
    type ProviderEvent = 'connect' | 'disconnect' | 'accountChanged';

    const listenerRegistry: Record<ProviderEvent, Set<Handler>> = {
      connect: new Set(),
      disconnect: new Set(),
      accountChanged: new Set(),
    };

    const isProviderEvent = (event: string): event is ProviderEvent =>
      event === 'connect' || event === 'disconnect' || event === 'accountChanged';

    class StoryWebSocket {
      public onopen: ((event: Event) => void) | null = null;
      public onmessage: ((event: MessageEvent) => void) | null = null;
      public onerror: ((event: Event) => void) | null = null;
      public onclose: ((event: CloseEvent) => void) | null = null;
      public readyState = 1;
      public url: string;

      constructor(url: string) {
        this.url = url;
        setTimeout(() => {
          this.onopen?.(new Event('open'));
          setTimeout(() => {
            const message = new MessageEvent('message', {
              data: JSON.stringify(mockAlertPayload),
            });
            this.onmessage?.(message);
          }, 400);
        }, 100);
      }

      send(): void {}

      close(): void {
        this.readyState = 3;
        this.onclose?.(new CloseEvent('close'));
      }
    }

    window.WebSocket = StoryWebSocket as unknown as typeof WebSocket;

    const mockProvider = {
      isPhantom: true,
      connect: async () => {
        const response = {
          publicKey: {
            toString: () => 'DemoWallet11111111111111111111111111111111',
          },
        };
        listenerRegistry.connect.forEach((listener) => listener(response.publicKey));
        return response;
      },
      disconnect: async () => {
        listenerRegistry.disconnect.forEach((listener) => listener());
      },
      on: (event: string, handler: Handler) => {
        if (isProviderEvent(event)) {
          listenerRegistry[event].add(handler);
        }
      },
      off: (event: string, handler: Handler) => {
        if (isProviderEvent(event)) {
          listenerRegistry[event].delete(handler);
        }
      },
      removeListener: (event: string, handler: Handler) => {
        if (isProviderEvent(event)) {
          listenerRegistry[event].delete(handler);
        }
      },
    };

    (window as any).solana = mockProvider;

    if (!navigator.mediaDevices) {
      (navigator as any).mediaDevices = {};
    }
    if (navigator.mediaDevices) {
      navigator.mediaDevices.getUserMedia = async () => new MediaStream();
    }

    process.env.NEXT_PUBLIC_BACKEND_URL = 'http://localhost:8000';

    return () => {
      window.WebSocket = originalWebSocket;
      (window as any).solana = originalSolana;
      if (navigator.mediaDevices && originalMediaDevices) {
        navigator.mediaDevices.getUserMedia = originalMediaDevices.bind(navigator.mediaDevices);
      }
      process.env.NEXT_PUBLIC_BACKEND_URL = originalBackendUrl;
    };
  }, []);

  return <>{children}</>;
};

const meta: Meta<typeof Stream> = {
  title: 'Dashcam/Stream',
  component: Stream,
  decorators: [
    (Story) => (
      <MockEnvironment>
        <Story />
      </MockEnvironment>
    ),
  ],
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof Stream>;

export const Default: Story = {
  args: {},
  name: 'Live Stream Demo',
};

