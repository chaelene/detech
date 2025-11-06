"""Backend integration helpers for the swarm refinement pipeline."""

from __future__ import annotations

from copy import deepcopy
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import types

from redis.asyncio import Redis

SWARM_SRC = Path(__file__).resolve().parents[3] / "swarm-agents" / "src"
if str(SWARM_SRC) not in sys.path:
    sys.path.append(str(SWARM_SRC))

_AGENTS_MODULE_PATH = SWARM_SRC / "agents.py"
if not _AGENTS_MODULE_PATH.is_file():
    raise RuntimeError(f"Expected swarm agents module at {_AGENTS_MODULE_PATH}")

swarm_pkg = types.ModuleType("swarm_agents")
swarm_pkg.__path__ = [str(SWARM_SRC)]
sys.modules.setdefault("swarm_agents", swarm_pkg)

_spec = importlib_util.spec_from_file_location(
    "swarm_agents.agents", str(_AGENTS_MODULE_PATH), submodule_search_locations=[str(SWARM_SRC)]
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load swarm agents module")

_agents_module = importlib_util.module_from_spec(_spec)
sys.modules["swarm_agents.agents"] = _agents_module
_spec.loader.exec_module(_agents_module)

InterpreterAgent = _agents_module.InterpreterAgent
RefinerAgent = _agents_module.RefinerAgent


class SwarmCoordinator:
    """Thin orchestrator that bridges backend alerts with swarm agents."""

    def __init__(
        self,
        *,
        redis: Redis,
        mock_mode: bool = False,
        refiner: Optional[RefinerAgent] = None,
        interpreter: Optional[InterpreterAgent] = None,
    ) -> None:
        self.redis = redis
        self.mock_mode = mock_mode
        if mock_mode:
            self.refiner = refiner
            self.interpreter = interpreter
        else:
            self.refiner = refiner or RefinerAgent(redis=redis)
            self.interpreter = interpreter or InterpreterAgent(redis=redis)

    async def process_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.mock_mode:
            return self._mock_process(payload)
        refined = await self.refiner.refine(payload)
        interpreted = await self.interpreter.interpret(refined)

        message = deepcopy(payload)
        message.setdefault("swarm", {})
        message["swarm"].update(
            {
                "refined": refined,
                "commands": interpreted.get("commands", []),
            }
        )
        message.setdefault("commands", interpreted.get("commands", []))
        message.setdefault("threat_level", refined.get("threat_level"))
        message.setdefault("anomaly_score", refined.get("anomaly_score"))
        return message

    def _mock_process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = deepcopy(payload)

        detection = payload.get("detection") or {}
        objects = detection.get("objects") or []
        confidences = [
            float(obj.get("confidence", 0))
            for obj in objects
            if isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float))
        ]
        baseline_conf = max(confidences) if confidences else 0.65
        refined_accuracy = round(min(0.98, max(0.72, baseline_conf + 0.12)), 4)
        anomaly_score = round(min(0.99, max(0.5, refined_accuracy - 0.04)), 4)
        threat_level = "critical" if refined_accuracy >= 0.9 else "high" if refined_accuracy >= 0.82 else "elevated"

        evolution = {
            "baseline_confidence": round(baseline_conf, 4),
            "refined_accuracy": refined_accuracy,
            "delta": round(refined_accuracy - baseline_conf, 4),
        }

        message.setdefault("swarm", {})
        message["swarm"].update(
            {
                "refined": {
                    "accuracy": refined_accuracy,
                    "confidence": refined_accuracy,
                    "anomaly_score": anomaly_score,
                    "threat_level": threat_level,
                },
                "commands": message.get("commands") or (["dispatch-drone"] if refined_accuracy >= 0.8 else ["log-event"]),
                "evolution": evolution,
            }
        )

        message["commands"] = message["swarm"]["commands"]
        message["swarm_confidence"] = refined_accuracy
        message["threat_level"] = threat_level
        message["anomaly_score"] = anomaly_score
        message.setdefault("metadata", {})
        message["metadata"].setdefault("swarm_evolution", evolution)
        return message

