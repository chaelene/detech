"""Tests for swarm agents orchestrating refinement and interpretation."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest


SWARM_SRC = Path(__file__).resolve().parents[1] / "swarm-agents" / "src"
if str(SWARM_SRC) not in sys.path:
    sys.path.append(str(SWARM_SRC))

from agents import InterpreterAgent, RefinerAgent  # noqa: E402

from backend.app.services.swarm import SwarmCoordinator  # noqa: E402


class DummyChain:
    async def ainvoke(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {
            "threat_level": "high",
            "anomaly_score": 0.92,
            "rationale": "Dummy chain response",
        }


@dataclass
class InMemoryRedis:
    storage: Dict[str, List[str]] = field(default_factory=dict)
    published: Dict[str, List[str]] = field(default_factory=dict)

    async def rpush(self, key: str, value: str) -> None:
        self.storage.setdefault(key, []).append(value)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        entries = self.storage.get(key, [])
        if not entries:
            return
        length = len(entries)
        start_idx = start if start >= 0 else max(length + start, 0)
        end_idx = end if end >= 0 else length + end
        end_idx = min(end_idx, length - 1)
        if start_idx > end_idx:
            self.storage[key] = []
        else:
            self.storage[key] = entries[start_idx : end_idx + 1]

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        entries = self.storage.get(key, [])
        if not entries:
            return []
        length = len(entries)
        start_idx = start if start >= 0 else max(length + start, 0)
        end_idx = end if end >= 0 else length + end
        end_idx = min(end_idx, length - 1)
        if start_idx > end_idx:
            return []
        return entries[start_idx : end_idx + 1]

    async def publish(self, channel: str, message: str) -> None:
        self.published.setdefault(channel, []).append(message)


def _make_detection(ts: str, gesture_conf: float = 0.85) -> Dict[str, Any]:
    return {
        "session_id": "session-123",
        "timestamp": ts,
        "objects": [{"name": "person", "confidence": 0.8}],
        "gestures": [{"name": "wave", "confidence": gesture_conf}],
    }


@pytest.mark.asyncio
async def test_refiner_produces_llm_enhanced_output_after_three_frames():
    redis = InMemoryRedis()
    refiner = RefinerAgent(redis=redis, chain=DummyChain(), window_size=3)

    for index in range(3):
        refined = await refiner.refine(_make_detection(f"2024-06-0{index}T00:00:00Z"))

    assert refined["threat_level"] == "high"
    assert refined["anomaly_score"] == 0.92
    assert "swarm:refined" in redis.published


@pytest.mark.asyncio
async def test_interpreter_maps_wave_gesture_to_rule_action():
    redis = InMemoryRedis()
    interpreter = InterpreterAgent(redis=redis, chain=DummyChain())

    refined_alert = {
        "id": "alert-1",
        "session_id": "session-123",
        "threat_level": "medium",
        "anomaly_score": 0.55,
        "votes": {"gestures": {"wave": {"votes": 2, "avg_confidence": 0.9}}},
    }

    actions = await interpreter.interpret(refined_alert)

    assert actions["commands"], "Expected interpreter to emit at least one command"
    command = actions["commands"][0]
    assert command["target"] == "alexa_mock"
    assert command["gesture"] == "wave"
    assert "swarm:actions" in redis.published


@pytest.mark.asyncio
async def test_swarm_coordinator_enriches_alert_payload():
    redis = InMemoryRedis()
    refiner = RefinerAgent(redis=redis, chain=DummyChain(), window_size=3)
    interpreter = InterpreterAgent(redis=redis, chain=DummyChain())
    coordinator = SwarmCoordinator(redis=redis, refiner=refiner, interpreter=interpreter)

    enriched = None
    for index in range(3):
        enriched = await coordinator.process_alert(
            _make_detection(f"2024-06-0{index}T00:00:00Z", gesture_conf=0.8 + index * 0.05)
        )

    assert enriched is not None
    swarm_block = enriched["swarm"]
    assert swarm_block["refined"]["threat_level"] == "high"
    assert swarm_block["commands"], "Commands should be available after interpretation"

