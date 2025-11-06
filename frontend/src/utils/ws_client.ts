export interface WsClient<TPayload = unknown> {
  /** Underlying WebSocket instance */
  socket: WebSocket;
  /** Close the websocket connection */
  close: (code?: number, reason?: string) => void;
  /** Send a JSON serialisable payload */
  sendJson: (payload: unknown) => void;
}

export interface WsClientOptions<TPayload = unknown> {
  url: string;
  onOpen?: (event: Event) => void;
  onMessage?: (payload: TPayload) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  /** Disable JSON parsing on incoming messages */
  raw?: boolean;
  /** Explicit reconnection backoff sequence in milliseconds */
  retryDelays?: number[];
  /** Maximum number of automatic reconnect attempts. Defaults to unlimited. */
  maxRetries?: number;
}

const DEFAULT_RETRY_DELAYS = [500, 1_000, 2_000, 5_000];

export function createWsClient<TPayload = unknown>(options: WsClientOptions<TPayload>): WsClient<TPayload> {
  const { url, onOpen, onMessage, onError, onClose, raw = false, retryDelays, maxRetries } = options;

  const delays = (retryDelays && retryDelays.length > 0 ? retryDelays : DEFAULT_RETRY_DELAYS).map((value) =>
    Math.max(100, Math.floor(value))
  );
  const retriesLimit = typeof maxRetries === 'number' && Number.isFinite(maxRetries) ? Math.max(0, maxRetries) : Infinity;

  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectAttempts = 0;
  let manualClose = false;

  const attachHandlers = (ws: WebSocket) => {
    if (onOpen) {
      ws.addEventListener('open', (event) => {
        reconnectAttempts = 0;
        onOpen(event);
      });
    }
    if (onError) {
      ws.addEventListener('error', onError);
    }
    if (onClose) {
      ws.addEventListener('close', onClose);
    }
    if (onMessage) {
      ws.addEventListener('message', (event) => {
        if (raw) {
          onMessage(event as unknown as TPayload);
          return;
        }
        try {
          const parsed = JSON.parse(event.data) as TPayload;
          onMessage(parsed);
        } catch (error) {
          console.error('ws_client: failed to parse JSON payload', error);
        }
      });
    }

    ws.addEventListener('close', (event) => {
      if (manualClose) {
        return;
      }
      if (event.code === 1000 || reconnectAttempts >= retriesLimit) {
        return;
      }
      const delay = delays[Math.min(reconnectAttempts, delays.length - 1)];
      reconnectAttempts += 1;
      reconnectTimer = setTimeout(connect, delay);
    });
  };

  let client: WsClient<TPayload>;

  const connect = () => {
    if (typeof WebSocket === 'undefined') {
      throw new Error('WebSocket API unavailable in this environment');
    }
    const ws = new WebSocket(url);
    socket = ws;
    client.socket = ws;
    attachHandlers(ws);
  };

  client = {
    socket: null as unknown as WebSocket,
    close: (code?: number, reason?: string) => {
      manualClose = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      socket?.close(code, reason);
    },
    sendJson: (payload: unknown) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }
      try {
        socket.send(JSON.stringify(payload));
      } catch (error) {
        console.error('ws_client: failed to send JSON payload', error);
      }
    },
  };

  connect();
  return client;
}

export type { WsClientOptions as WebSocketClientOptions };

