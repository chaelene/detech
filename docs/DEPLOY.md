# DETECH Deployment Notes

The project ships with a docker compose simulation and CPU-friendly Jetson mock.
This guide covers a minimal production-style rollout on a VPS plus the steps to
run the detector on a Jetson (or x86 dev kit) under systemd.

## 1. Prerequisites

- Docker Engine ≥ 24 and docker compose plugin.
- Node.js 20+ (only required if you plan to build the frontend outside Docker).
- Python 3.10 (already encapsulated inside the containers).
- Redis 7+ and Mosquitto 2.0 (provided by compose stack).
- Optional: `obs-cli` and `ffmpeg` for demo recordings (see `docs/demo.sh`).

## 2. VPS deployment (compose stack)

```bash
git clone https://github.com/<org>/detech.git
cd detech

# Provide secrets via .env (example values shown)
cat <<'EOF' > .env
SOLANA_RPC_URL=https://api.devnet.solana.com
OPENAI_API_KEY=sk-...
X402_AGENT_PRIVATE_KEY=
EOF

# Bring up the simulation stack (backend, frontend, redis, mqtt, swarm, jetson-mock)
docker compose up -d --build

# Tail logs while validating
docker compose logs -f backend swarm jetson
```

> **Health checks:**
> - Backend: `curl http://localhost:8000/health`
> - Frontend: open `http://localhost:3000`
> - MQTT: `mosquitto_sub -h localhost -t detech/alerts -C 1`

For deployments that should persist container state, add a reverse proxy (Caddy,
Nginx, Traefik) in front of the frontend/backend and map TLS certificates.

## 3. Jetson / edge node service

The repository contains the full GPU pipeline in `jetson-edge/src/detector.py`.
To run it headlessly on boot, place the service under systemd (replace
`/opt/detech` with your clone path).

`/etc/systemd/system/detech-detector.service`

```
[Unit]
Description=DETECH Jetson detector service
After=network-online.target mosquitto.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/detech
Environment=MQTT_BROKER_HOST=backend.internal
Environment=MQTT_BROKER_PORT=1883
Environment=FRAMES_TOPIC=detech/frames
Environment=ALERTS_TOPIC=detech/alerts
Environment=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64
ExecStart=/usr/bin/python3 -m jetson-edge.src.detector --broker-host ${MQTT_BROKER_HOST} --frames-topic ${FRAMES_TOPIC} --alerts-topic ${ALERTS_TOPIC} --verbose
Restart=on-failure
RestartSec=5
User=detech

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now detech-detector.service
sudo journalctl -fu detech-detector.service
```

### CPU development mode

On non-Jetson hosts you can run the deterministic CPU mock used by the compose
stack:

```bash
python -m jetson-edge.src.mock_cpu_detector \
  --broker-host localhost \
  --broker-port 1883
```

## 4. Recording demos

The helper script `docs/demo.sh` automates health checks, starts a mock stream,
and records the UI via `ffmpeg` or OBS. Example (Linux/X11 capture):

```bash
MODE=ffmpeg \
FFMPEG_INPUT_ARGS="-f x11grab -video_size 1920x1080 -i :0.0" \
./docs/demo.sh --duration 60 --output ./docs/demo.mp4
```

For OBS (requires `obs-cli`):

```bash
MODE=obs OBS_PASSWORD=secret ./docs/demo.sh --duration 90
```

## 5. Operational tips

- **Latency metrics** appear in backend logs as `Alert delivered` with
  `latency_ms`. They are also attached to the alert payload (`metrics.ingest_latency_ms`).
- **Swarm refinement** can run in `mock` mode (default in compose). Set
  `SWARM_MOCK_MODE=0` and provide a real `OPENAI_API_KEY` to enable the LangChain
  agents in production.
- **x402 billing** uses the mock transactor unless `X402_MOCK_TRANSFERS=0` and an
  agent key are configured. In mock mode a synthetic signature is returned.

Stay synced with the compose stack (`docker compose pull --ignore-pull-failures`)
and review `tests/integration/` for regression coverage that should pass before
rolling out updates.

