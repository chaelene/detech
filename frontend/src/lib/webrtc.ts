import { useState, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function useWebRTC() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(async () => {
    try {
      const newSocket = io(BACKEND_URL, {
        transports: ['websocket'],
      });

      newSocket.on('connect', () => {
        console.log('Connected to backend');
        setIsConnected(true);
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from backend');
        setIsConnected(false);
      });

      newSocket.on('alert', (data) => {
        // TODO: Handle incoming alerts from swarm
        console.log('Received alert:', data);
      });

      setSocket(newSocket);
    } catch (error) {
      console.error('Failed to connect to backend:', error);
      throw error;
    }
  }, []);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.disconnect();
      setSocket(null);
      setIsConnected(false);
    }
  }, [socket]);

  const sendStream = useCallback(async (stream: MediaStream) => {
    if (!socket || !isConnected) {
      throw new Error('Not connected to backend');
    }

    // TODO: Implement WebRTC peer connection with Mediasoup
    // This will establish a peer connection and send the video stream
    // to the backend which will forward it to Jetson edge device
    
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });

    stream.getTracks().forEach((track) => {
      pc.addTrack(track, stream);
    });

    // TODO: Exchange SDP offers/answers with backend
    // TODO: Handle ICE candidates
    socket.emit('stream-start', { streamId: 'user-stream' });
  }, [socket, isConnected]);

  return {
    connect,
    disconnect,
    sendStream,
    isConnected,
  };
}
