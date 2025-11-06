export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ObjectDetection {
  name: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface GestureDetection {
  name: string;
  confidence: number;
  landmarks?: number[][];
}

export interface Detection {
  timestamp: number;
  objects: ObjectDetection[];
  gestures: GestureDetection[];
}

export interface Alert {
  id: string;
  sessionId?: string;
  timestamp: number;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  detection?: Detection;
  source: 'jetson' | 'swarm';
  paymentStatus?: 'pending' | 'confirmed' | 'failed';
  paymentTxId?: string;
  paymentAmount?: number;
  swarmConfidence?: number;
  commands?: string[];
  threatLevel?: string;
  anomalyScore?: number;
  x402?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface StreamOfferPayload {
  session_id: string;
  wallet_pubkey: string;
  sdp: string;
  ice_candidates?: RTCIceCandidateInit[];
  metadata?: Record<string, unknown>;
}

export interface StreamAnswerPayload {
  session_id: string;
  answer_sdp: string;
}

export interface AlertSocketPayload extends Record<string, unknown> {
  session_id: string;
  alert_id?: string;
  label?: string;
  severity?: Alert['severity'];
  description?: string;
  detection?: Detection;
  commands?: string[];
  payment_status?: Alert['paymentStatus'];
  payment_tx_id?: string;
  payment_amount?: number;
  swarm_confidence?: number;
  threat_level?: string;
  anomaly_score?: number;
  x402?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  source?: Alert['source'];
  timestamp?: number;
}

export interface StreamConfig {
  streamId: string;
  resolution: {
    width: number;
    height: number;
  };
  fps: number;
  codec: 'vp8' | 'vp9' | 'h264';
}
