# DETECH API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement proper authentication.

## WebSocket Events (Socket.IO)

### Client → Server

#### `stream-start`
Start a new stream.

**Payload**:
```json
{
  "streamId": "user-stream-123"
}
```

#### `stream-stop`
Stop the current stream.

**Payload**: None

### Server → Client

#### `connected`
Connection confirmation.

**Payload**:
```json
{
  "sid": "socket_id"
}
```

#### `alert`
Incoming alert from swarm.

**Payload**:
```json
{
  "id": "alert-123",
  "timestamp": 1699123456.789,
  "type": "object_detection",
  "severity": "high",
  "description": "Person detected with weapon",
  "detection": {
    "objects": [...],
    "gestures": [...]
  },
  "payment_status": "confirmed",
  "payment_tx_id": "5j7s8K9...",
  "payment_amount": 0.01
}
```

## REST API

### Health Check

#### `GET /health`

Check service health.

**Response**:
```json
{
  "status": "healthy",
  "streaming_service": true,
  "solana_service": true
}
```

### Payment

#### `POST /api/v1/payment`

Create x402 payment for alert.

**Request Body**:
```json
{
  "recipient": "Solana_address",
  "amount_usdc": 0.01,
  "alert_id": "alert-123"
}
```

**Response**:
```json
{
  "tx_signature": "transaction_signature",
  "status": "pending"
}
```

#### `GET /api/v1/payment/{tx_signature}`

Verify payment transaction.

**Response**:
```json
{
  "tx_signature": "transaction_signature",
  "verified": true
}
```

### Streams

#### `GET /api/v1/streams`

List active streams.

**Response**:
```json
{
  "streams": [
    {
      "stream_id": "user-stream-123",
      "client_id": "socket_id",
      "status": "active"
    }
  ]
}
```
