# DETECH

Edge-to-cloud AI vigilance system combining:

- Jetson edge detector (YOLO11 + MediaPipe) publishing MQTT alerts
- FastAPI backend coordinating WebRTC ingest, swarm agents, and Solana/X402
- LangChain swarm agents refining detections
- Next.js frontend for live alerts and wallet interactions

## Getting Started

Install dependencies per component:

    pip install -r backend/requirements.txt
    pip install -r swarm-agents/requirements.txt
    npm install --prefix frontend

Then launch the detector/backend/frontend as needed.
