# DETECH Architecture

## System Overview

DETECH is designed as a decentralized AI Vigilance Swarm with a clear separation between edge computing (Jetson) and cloud services (VPS). The architecture emphasizes:

1. **Edge-Only CV Processing**: Jetson handles all raw computer vision (YOLO11 + MediaPipe)
2. **Cloud-Based Swarm Intelligence**: VPS runs LangChain agents for collaborative analysis
3. **Decentralized Payments**: Solana x402 protocol for micro-payments
4. **Browser-First Access**: No app installation required

## Component Architecture

### Frontend (Browser)
- **Role**: User interface and stream capture
- **Technology**: Next.js 14, React, WebRTC
- **Responsibilities**:
  - Capture video stream via WebRTC
  - Connect Solana wallet via WalletConnect
  - Display alerts and payment status
  - Handle user interactions

### Backend (VPS)
- **Role**: Orchestration and coordination
- **Technology**: FastAPI, Mediasoup, Socket.IO
- **Responsibilities**:
  - Relay WebRTC streams to Jetson
  - Coordinate swarm agents
  - Process x402 payments
  - Broadcast alerts to frontend

### Jetson Edge (Orin Nano)
- **Role**: Raw CV detection only
- **Technology**: YOLO11, MediaPipe, OpenCV
- **Responsibilities**:
  - Object detection (YOLO11)
  - Gesture recognition (MediaPipe)
  - Publish detections via MQTT
  - **NO** alert refinement or payment processing

### Swarm Agents (VPS)
- **Role**: Collaborative AI analysis
- **Technology**: LangChain, OpenAI
- **Responsibilities**:
  - Refine raw detections into alerts
  - Analyze context and patterns
  - Generate recommendations
  - Store analysis in Redis

### Shared Utilities
- **Role**: Common code and schemas
- **Components**:
  - Pydantic schemas for data validation
  - Solana x402 payment helpers
  - Common utilities

## Data Flow

### Stream Flow
```
Browser → Backend (Mediasoup) → Jetson (MQTT) → Backend → Swarm → Backend → Browser
```

### Detection Flow
```
Jetson (YOLO11/MediaPipe) → MQTT → Backend → Swarm (LangChain) → Redis → Backend → Browser
```

### Payment Flow
```
Swarm Alert → Backend → Solana (x402) → Backend → Browser
```

## Message Protocols

### MQTT Topics
- `detech/detections`: Raw detections from Jetson
- `detech/alerts`: Refined alerts from Swarm

### Redis Keys
- `detection:{timestamp}`: Stored detections
- `alert:{id}`: Stored alerts
- `stream:{id}`: Stream metadata

## Security Considerations

1. **Wallet Security**: Private keys stored securely (never in code)
2. **WebRTC Security**: Use TLS/DTLS for encrypted streams
3. **MQTT Security**: Use authentication and TLS
4. **API Security**: Implement rate limiting and authentication
5. **Solana Security**: Use proper key management and transaction validation

## Scalability

### Horizontal Scaling
- Backend: Stateless, can scale horizontally
- Swarm Agents: Can run multiple instances
- Redis: Use Redis Cluster for scaling

### Vertical Scaling
- Jetson: Optimize model inference (TensorRT)
- Backend: Increase CPU/memory for WebRTC handling

## Monitoring

### Key Metrics
- Stream latency (Browser → Jetson → Backend)
- Detection rate (frames per second)
- Alert generation time
- Payment confirmation time
- Swarm agent response time

### Logging
- Structured logging (JSON format)
- Centralized log aggregation
- Error tracking and alerting

## Future Enhancements

1. **Federated Learning**: Share model updates across Jetson devices
2. **Multi-Jetson Support**: Coordinate multiple edge devices
3. **Advanced Gesture Commands**: Expand gesture vocabulary
4. **Real-time Collaboration**: Multiple users viewing same stream
5. **Blockchain Oracles**: External data integration for context
