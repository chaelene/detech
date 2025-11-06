"""Utility script to exercise the detector pipeline with a sample video.

The script reuses the production `FrameProcessor`, simulating the MQTT ingress by
encoding frames to base64 and then validating the resulting alert payloads
against a JSON schema. This provides a lightweight regression harness to ensure
the edge pipeline remains well-formed before deployment.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
from jsonschema import ValidationError, validate

from utils.frame_processor import FrameProcessor


ALERT_SCHEMA = {
    "type": "object",
    "required": ["timestamp", "frame_index", "objects", "gestures"],
    "properties": {
        "timestamp": {"type": "string"},
        "frame_index": {"type": "integer", "minimum": 0},
        "source": {"type": "string"},
        "metadata": {"type": "object"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["track_id", "label", "confidence", "bbox", "event"],
                "properties": {
                    "track_id": {"type": "integer", "minimum": 0},
                    "label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "event": {"type": "string", "enum": ["enter", "update", "exit"]},
                    "bbox": {
                        "type": "object",
                        "required": ["x1", "y1", "x2", "y2"],
                        "properties": {
                            "x1": {"type": "integer", "minimum": 0},
                            "y1": {"type": "integer", "minimum": 0},
                            "x2": {"type": "integer", "minimum": 0},
                            "y2": {"type": "integer", "minimum": 0},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "gestures": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "command": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": True,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector pipeline against a sample video")
    parser.add_argument("--video", default="video.mp4", help="Path to sample MP4 video")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO11 weights to use")
    parser.add_argument("--engine", default=None, help="Optional TensorRT engine path")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum frames to process")
    parser.add_argument("--min-alerts", type=int, default=1, help="Minimum number of alerts to validate")
    parser.add_argument("--verbose", action="store_true", help="Print every validated alert payload")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Sample video not found: {video_path}")

    processor = FrameProcessor(model_path=args.model, engine_path=args.engine)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        processor.close()
        raise SystemExit(f"Failed to open video: {video_path}")

    alerts_validated = 0
    processed_frames = 0

    try:
        while processed_frames < args.max_frames:
            processed_frames += 1

            ret, frame = cap.read()
            if not ret:
                break

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            alert = processor.process_base64_frame(frame_b64)
            if not alert:
                continue

            try:
                validate(instance=alert, schema=ALERT_SCHEMA)
            except ValidationError as exc:  # pragma: no cover - validation path failures should surface loudly
                raise AssertionError(f"Alert payload failed schema validation: {exc.message}") from exc

            alerts_validated += 1
            if args.verbose:
                print(json.dumps(alert, indent=2))

            if alerts_validated >= args.min_alerts:
                break

    finally:
        cap.release()
        processor.close()

    if alerts_validated < args.min_alerts:
        raise AssertionError(
            "No alerts satisfied the schema validation. Verify that the sample video contains the target classes or gestures."
        )

    print(f"Validated {alerts_validated} alert(s) against schema from {processed_frames} frame(s).")


if __name__ == "__main__":
    main(sys.argv[1:])

