"""Reusable LangChain chains for the swarm refinement pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def _select_provider(provider: Optional[str]) -> str:
    env_provider = (
        provider
        or os.getenv("SWARM_REFINER_PROVIDER")
        or os.getenv("SWARM_LLM_PROVIDER")
        or "openai"
    )
    return env_provider.lower()


def _resolve_llm(provider: str, model: Optional[str], temperature: float):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-3-5-sonnet-20240620", temperature=temperature)

    if provider == "xai":
        try:
            from langchain_xai import ChatXAI
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("langchain-xai is not installed") from exc

        return ChatXAI(model=model or "xAI-Grok-2", temperature=temperature)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)


def build_refiner_chain(
    *, provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.1
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Create a chain that scores anomaly severity for a bundle of frames."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are part of a swarm of safety agents. "
                    "Given a bundle of detections, output JSON with keys "
                    "`threat_level` (one of low, medium, high), `anomaly_score` (0-1 float), "
                    "and `rationale` (short string)."
                ),
            ),
            (
                "human",
                (
                    "Frame summary:\n{frame_summary}\n\n"
                    "Cross-agent consensus (JSON):\n{consensus_json}\n"
                    "Focus: refine this person detection for threat level."
                ),
            ),
        ]
    )

    parser = JsonOutputParser()
    provider_name = _select_provider(provider)
    llm = _resolve_llm(provider_name, model, temperature)
    return prompt | llm | parser


def build_interpreter_chain(
    *, provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.2
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Create a chain that suggests actions for gestures not covered by rules."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You translate crowd gestures into automation commands. "
                    "Respond with JSON containing `action`, `target`, and `reason`."
                ),
            ),
            (
                "human",
                (
                    "Gesture context:\n{gesture_context}\n\n"
                    "Threat level: {threat_level}\n"
                    "Anomaly score: {anomaly_score}"
                ),
            ),
        ]
    )

    parser = JsonOutputParser()
    provider_name = _select_provider(provider)
    llm = _resolve_llm(provider_name, model, temperature)
    return prompt | llm | parser

