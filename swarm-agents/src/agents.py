"""Core swarm agents coordinating refinement and interpretation."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from redis.asyncio import Redis

from .chains.refiner import build_interpreter_chain, build_refiner_chain


DEFAULT_WINDOW = 3
DEFAULT_TIMEOUT_MS = 100


def _normalize(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _sanitize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky fields while keeping semantic context for the swarm."""

    keep_keys = {"id", "timestamp", "session_id", "camera_id", "objects", "gestures", "tracklets"}
    sanitized: Dict[str, Any] = {}
    for key in keep_keys:
        if key in detection:
            sanitized[key] = detection[key]

    def _trim_collection(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed = []
        for item in items:
            trimmed.append(
                {
                    "name": item.get("name") or item.get("label"),
                    "confidence": _safe_float(item.get("confidence")),
                    "id": item.get("id") or item.get("track_id"),
                }
            )
        return trimmed

    if "objects" in sanitized:
        sanitized["objects"] = _trim_collection(sanitized.get("objects", []))
    if "gestures" in sanitized:
        sanitized["gestures"] = _trim_collection(sanitized.get("gestures", []))
    if "tracklets" in sanitized:
        sanitized["tracklets"] = _trim_collection(sanitized.get("tracklets", []))
    return sanitized


def _baseline_threat(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@dataclass
class RefinerAgent:
    """Aggregates frames and refines detections into anomaly scores."""

    redis: Redis
    chain: Optional[Any] = None
    window_size: int = DEFAULT_WINDOW
    refined_channel: str = "swarm:refined"
    buffer_prefix: str = "swarm:frames"
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    _chain_initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.chain is None:
            self.chain = build_refiner_chain()
        self._chain_initialized = True

    @property
    def _timeout_seconds(self) -> float:
        return self.timeout_ms / 1000.0

    def _source_key(self, detection: Dict[str, Any]) -> str:
        for key in ("session_id", "camera_id", "device_id", "id"):
            value = detection.get(key)
            if value:
                return str(value)
        return "global"

    async def refine(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        source_id = self._source_key(detection)
        buffer_key = f"{self.buffer_prefix}:{source_id}"

        await self.redis.rpush(buffer_key, json.dumps(_sanitize_detection(detection)))
        await self.redis.ltrim(buffer_key, -self.window_size, -1)
        raw_frames = await self.redis.lrange(buffer_key, -self.window_size, -1)

        frames = [json.loads(frame) if isinstance(frame, str) else json.loads(frame.decode()) for frame in raw_frames]

        consensus = self._build_consensus(frames)
        frame_summary = self._frame_summary(frames)
        baseline_score = self._baseline_score(consensus)

        llm_payload = {
            "frame_summary": frame_summary,
            "consensus_json": json.dumps(consensus),
        }

        llm_result: Dict[str, Any] = {}
        if len(frames) >= self.window_size and self._chain_initialized:
            try:
                llm_result = await asyncio.wait_for(
                    self.chain.ainvoke(llm_payload), timeout=self._timeout_seconds
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                llm_result = {}
            except Exception:  # pragma: no cover - defensive logging handled upstream
                llm_result = {}

        anomaly_score = _safe_float(llm_result.get("anomaly_score"), baseline_score)
        threat_level = _normalize(llm_result.get("threat_level")) or _baseline_threat(anomaly_score)
        rationale = llm_result.get("rationale") or "Consensus-based baseline refinement"

        latency_ms = (time.perf_counter() - start) * 1000
        refined_alert = {
            "id": detection.get("id") or detection.get("alert_id") or str(uuid.uuid4()),
            "session_id": detection.get("session_id"),
            "source": "refiner_agent",
            "window_size": self.window_size,
            "anomaly_score": round(anomaly_score, 3),
            "threat_level": threat_level,
            "rationale": rationale,
            "votes": consensus,
            "frames": frames,
            "latency_ms": round(latency_ms, 2),
        }

        await self.redis.publish(self.refined_channel, json.dumps(refined_alert))
        return refined_alert

    def _build_consensus(self, frames: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {"objects": {}, "gestures": {}}
        for frame in frames:
            for obj in frame.get("objects", []) or []:
                name = _normalize(obj.get("name")) or "unknown"
                entry = result["objects"].setdefault(name, {"votes": 0, "avg_confidence": 0.0})
                entry["votes"] += 1
                entry["avg_confidence"] += _safe_float(obj.get("confidence"))

            for gesture in frame.get("gestures", []) or []:
                name = _normalize(gesture.get("name")) or "unknown"
                entry = result["gestures"].setdefault(name, {"votes": 0, "avg_confidence": 0.0})
                entry["votes"] += 1
                entry["avg_confidence"] += _safe_float(gesture.get("confidence"))

        for category in ("objects", "gestures"):
            for stats in result[category].values():
                if stats["votes"]:
                    stats["avg_confidence"] = round(stats["avg_confidence"] / stats["votes"], 3)
        return result

    def _frame_summary(self, frames: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, frame in enumerate(frames, start=1):
            timestamp = frame.get("timestamp", "unknown")
            objects = frame.get("objects", []) or []
            gestures = frame.get("gestures", []) or []
            lines.append(
                f"Frame {idx} at {timestamp}: {len(objects)} objects, {len(gestures)} gestures"
            )
            for obj in objects:
                lines.append(
                    f"  object={obj.get('name', 'unknown')} conf={_safe_float(obj.get('confidence')):.2f}"
                )
            for gesture in gestures:
                lines.append(
                    f"  gesture={gesture.get('name', 'unknown')} conf={_safe_float(gesture.get('confidence')):.2f}"
                )
        return "\n".join(lines)

    def _baseline_score(self, consensus: Dict[str, Dict[str, Any]]) -> float:
        confidences: List[float] = []
        for category in consensus.values():
            for stats in category.values():
                confidences.append(_safe_float(stats.get("avg_confidence")))
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 3)


@dataclass
class InterpreterAgent:
    """Translates swarm consensus into downstream automation commands."""

    redis: Redis
    chain: Optional[Any] = None
    gesture_rules: Optional[Dict[str, Dict[str, Any]]] = None
    output_channel: str = "swarm:actions"
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    _chain_initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.chain is None:
            self.chain = build_interpreter_chain()
        if self.gesture_rules is None:
            self.gesture_rules = {
                "wave": {
                    "action": "relay",
                    "target": "alexa_mock",
                    "command": "announce_presence",
                },
                "stop": {
                    "action": "relay",
                    "target": "security_console",
                    "command": "trigger_lockdown",
                },
                "thumbs_up": {
                    "action": "relay",
                    "target": "operator_ui",
                    "command": "acknowledge_event",
                },
            }
        self._chain_initialized = True

    @property
    def _timeout_seconds(self) -> float:
        return self.timeout_ms / 1000.0

    async def interpret(self, refined_alert: Dict[str, Any]) -> Dict[str, Any]:
        commands: List[Dict[str, Any]] = []
        gestures_consensus = refined_alert.get("votes", {}).get("gestures", {})

        for gesture_name, stats in gestures_consensus.items():
            rule = self.gesture_rules.get(gesture_name)
            if rule:
                commands.append(
                    {
                        **rule,
                        "gesture": gesture_name,
                        "confidence": round(_safe_float(stats.get("avg_confidence")), 3),
                        "votes": stats.get("votes", 0),
                        "reason": f"Rule-based mapping for {gesture_name}",
                        "source": "interpreter_agent",
                    }
                )
                continue

            if not self._chain_initialized:
                continue

            payload = {
                "gesture_context": json.dumps({"gesture": gesture_name, "stats": stats}),
                "threat_level": refined_alert.get("threat_level", "unknown"),
                "anomaly_score": refined_alert.get("anomaly_score", 0),
            }
            llm_result: Dict[str, Any] = {}
            try:
                llm_result = await asyncio.wait_for(
                    self.chain.ainvoke(payload), timeout=self._timeout_seconds
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                llm_result = {}
            except Exception:  # pragma: no cover - defensive logging done upstream
                llm_result = {}
            if llm_result:
                commands.append(
                    {
                        "gesture": gesture_name,
                        "action": llm_result.get("action", "operator_review"),
                        "target": llm_result.get("target", "operator_console"),
                        "reason": llm_result.get("reason", "LLM suggested action"),
                        "confidence": round(_safe_float(stats.get("avg_confidence")), 3),
                        "votes": stats.get("votes", 0),
                        "source": "interpreter_agent",
                    }
                )

        actions_payload = {
            "session_id": refined_alert.get("session_id"),
            "alert_id": refined_alert.get("id"),
            "commands": commands,
            "source": "interpreter_agent",
        }

        await self.redis.publish(self.output_channel, json.dumps(actions_payload))
        return actions_payload

    def register_gesture_rule(self, gesture: str, rule: Dict[str, Any]) -> None:
        self.gesture_rules[_normalize(gesture)] = rule


