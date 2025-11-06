"""Frame processing utilities for DETECH Jetson edge pipeline.

This module handles the core perception pipeline on the Jetson Orin Nano:

- Base64 frame decoding with robust error handling
- YOLO11 nano inference accelerated through CUDA / TensorRT when available
- Simple Kalman filter based tracking to detect enter / exit events
- MediaPipe Hands gesture classification (thumbs up -> command trigger)
- Packaging detections into lightweight JSON payloads suitable for MQTT alerts

All heavy lifting (YOLO + gesture classification) is executed on-device to ensure
raw detections never leave the edge. The resulting enriched payloads contain
only the metadata required for downstream vigilance on the broker / cloud side.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)


class FrameDecodeError(Exception):
    """Raised when a base64 payload cannot be decoded into a valid frame."""


@dataclass
class Track:
    """Internal representation of a tracked object."""

    track_id: int
    label: str
    kalman: cv2.KalmanFilter
    confidence: float
    last_bbox: Tuple[float, float, float, float]  # xyxy order
    last_update: float
    missed_frames: int = 0
    has_entered: bool = False
    _prediction_cached: bool = field(default=False, init=False, repr=False)
    _cached_prediction: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def predict(self) -> Tuple[float, float, float, float]:
        """Predict the next bounding box using the Kalman filter."""

        if not self._prediction_cached:
            self._cached_prediction = self.kalman.predict()
            self._prediction_cached = True
        state = self._cached_prediction[:4].reshape(-1)
        return _cxcywh_to_xyxy(state)

    def update(self, bbox_xyxy: Tuple[float, float, float, float], confidence: float, timestamp: float) -> None:
        """Correct the Kalman filter with the latest measurement."""

        measurement = np.array(_xyxy_to_cxcywh(bbox_xyxy), dtype=np.float32).reshape(4, 1)
        self.kalman.correct(measurement)
        self.last_bbox = bbox_xyxy
        self.confidence = confidence
        self.last_update = timestamp
        self.missed_frames = 0
        self._prediction_cached = False

    def commit_prediction(self) -> None:
        """Persist the last prediction as the current state when no measurement is available."""

        if self._prediction_cached and self._cached_prediction is not None:
            state = self._cached_prediction[:4].reshape(-1)
            bbox = _cxcywh_to_xyxy(state)
            self.last_bbox = bbox
            self._prediction_cached = False


def _xyxy_to_cxcywh(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def _cxcywh_to_xyxy(state: Iterable[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = state
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def _clip_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, int]:
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


class FrameProcessor:
    """Encapsulates frame decoding, detection, tracking, and gesture handling."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        engine_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 6,
        target_fps: float = 15.0,
        allowed_classes: Optional[Iterable[str]] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max(1, max_missed_frames)
        self.delta_t = 1.0 / max(target_fps, 1.0)
        self.allowed_classes = set(allowed_classes or ("person", "car"))
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_index = 0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = self._load_model(model_path, engine_path)
        self.model_overrides = {
            "conf": self.conf_threshold,
            "imgsz": 640,
            "device": self.device,
            "half": self.device.startswith("cuda"),
        }

        self.class_name_map = self._resolve_class_map()
        self.allowed_class_ids = {idx for idx, name in self.class_name_map.items() if name in self.allowed_classes}
        if not self.allowed_class_ids:
            raise ValueError("Configured YOLO model does not expose the required classes: %s" % self.allowed_classes)

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.gesture_label = "thumbs_up"

        LOGGER.info(
            "FrameProcessor initialised | device=%s | TensorRT=%s | allowed_classes=%s",
            self.device,
            str(engine_path is not None and Path(engine_path).exists()),
            sorted(self.allowed_classes),
        )

    def _load_model(self, model_path: str, engine_path: Optional[str]) -> YOLO:
        """Load YOLO model with optional TensorRT engine for acceleration."""

        weights_path: Optional[Path] = None
        if engine_path:
            candidate = Path(engine_path)
            if candidate.exists():
                weights_path = candidate
        if weights_path is None:
            weights_path = Path(model_path)

        try:
            model = YOLO(str(weights_path))
        except Exception as exc:  # pragma: no cover - ultralytics handles downloads
            LOGGER.error("Failed to load YOLO weights from %s: %s", weights_path, exc)
            raise

        # Attempt to fuse layers for better throughput (safe no-op if unsupported)
        try:
            model.fuse()
        except Exception:  # pragma: no cover - fuse not always available
            LOGGER.debug("Skipping layer fusion for model %s", weights_path)

        return model

    def _resolve_class_map(self) -> Dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(idx): str(name) for idx, name in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def decode_frame(self, frame_b64: str) -> np.ndarray:
        """Decode a base64 JPEG payload into an OpenCV BGR frame."""

        if not frame_b64:
            raise FrameDecodeError("Empty frame payload")
        try:
            jpeg_bytes = base64.b64decode(frame_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FrameDecodeError(f"Invalid base64 payload: {exc}") from exc

        frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameDecodeError("Failed to decode JPEG payload")
        return frame

    def process_base64_frame(self, frame_b64: str, timestamp: Optional[float] = None) -> Optional[Dict[str, object]]:
        """Decode and process a base64 encoded frame."""

        try:
            frame = self.decode_frame(frame_b64)
        except FrameDecodeError as exc:
            LOGGER.debug("Dropping malformed frame: %s", exc)
            return None
        return self.process_frame(frame, timestamp=timestamp)

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Optional[Dict[str, object]]:
        """Run detections and tracking on a decoded frame."""

        if frame is None or frame.size == 0:
            LOGGER.debug("Received empty frame for processing; ignoring")
            return None

        self.frame_index += 1
        now = timestamp or time.time()
        iso_timestamp = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        objects = self._detect_objects(frame)
        tracked_objects = self._update_tracks(objects, now, frame.shape)
        gestures = self._detect_gestures(frame)

        if not tracked_objects and not gestures:
            # Keep heartbeat minimal when nothing interesting detected
            return {
                "timestamp": iso_timestamp,
                "frame_index": self.frame_index,
                "objects": [],
                "gestures": [],
            }

        payload: Dict[str, object] = {
            "timestamp": iso_timestamp,
            "frame_index": self.frame_index,
            "objects": tracked_objects,
            "gestures": gestures,
        }
        return payload

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, object]]:
        results = self.model(frame, verbose=False, **self.model_overrides)
        detections: List[Dict[str, object]] = []

        for result in results:
            if not hasattr(result, "boxes"):
                continue
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.allowed_class_ids:
                    continue
                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": self.class_name_map.get(cls_id, str(cls_id)),
                        "confidence": confidence,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )

        return detections

    def _update_tracks(
        self,
        detections: List[Dict[str, object]],
        timestamp: float,
        frame_shape: Tuple[int, int, int],
    ) -> List[Dict[str, object]]:
        frame_height, frame_width = frame_shape[:2]
        outputs: List[Dict[str, object]] = []

        # Predict current positions for existing tracks
        predictions: Dict[int, Tuple[float, float, float, float]] = {}
        for track_id, track in self.tracks.items():
            predictions[track_id] = track.predict()

        assigned_tracks: Dict[int, int] = {}

        for det_idx, detection in enumerate(detections):
            det_bbox = detection["bbox"]
            best_track_id = None
            best_iou = 0.0

            for track_id, predicted_bbox in predictions.items():
                if track_id in assigned_tracks:
                    continue
                iou = _compute_iou(predicted_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_id]
                track.update(det_bbox, float(detection["confidence"]), timestamp)
                assigned_tracks[best_track_id] = det_idx
                event = "enter" if not track.has_entered else "update"
                track.has_entered = True
                outputs.append(
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(float(track.confidence), 4),
                        "bbox": _clip_bbox(track.last_bbox, frame_width, frame_height),
                        "event": event,
                    }
                )
            else:
                # Create a new track
                new_track = self._create_track(detection, timestamp)
                self.tracks[new_track.track_id] = new_track
                assigned_tracks[new_track.track_id] = det_idx
                outputs.append(
                    {
                        "track_id": new_track.track_id,
                        "label": new_track.label,
                        "confidence": round(float(new_track.confidence), 4),
                        "bbox": _clip_bbox(new_track.last_bbox, frame_width, frame_height),
                        "event": "enter",
                    }
                )

        # Handle tracks without current detections
        tracks_to_remove: List[int] = []
        for track_id, track in self.tracks.items():
            if track_id in assigned_tracks:
                continue
            track.commit_prediction()
            track.missed_frames += 1
            if track.missed_frames >= self.max_missed_frames and track.has_entered:
                outputs.append(
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(float(track.confidence), 4),
                        "bbox": _clip_bbox(track.last_bbox, frame_width, frame_height),
                        "event": "exit",
                    }
                )
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            self.tracks.pop(track_id, None)

        return outputs

    def _create_track(self, detection: Dict[str, object], timestamp: float) -> Track:
        bbox = detection["bbox"]
        kalman = cv2.KalmanFilter(8, 4)
        transition = np.eye(8, dtype=np.float32)
        for i in range(4):
            transition[i, i + 4] = self.delta_t
        kalman.transitionMatrix = transition

        measurement_matrix = np.zeros((4, 8), dtype=np.float32)
        measurement_matrix[0, 0] = 1.0
        measurement_matrix[1, 1] = 1.0
        measurement_matrix[2, 2] = 1.0
        measurement_matrix[3, 3] = 1.0
        kalman.measurementMatrix = measurement_matrix

        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kalman.errorCovPost = np.eye(8, dtype=np.float32)

        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        kalman.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)

        track = Track(
            track_id=self.next_track_id,
            label=str(detection["label"]),
            kalman=kalman,
            confidence=float(detection["confidence"]),
            last_bbox=bbox,
            last_update=timestamp,
        )
        self.next_track_id += 1
        return track

    # ------------------------------------------------------------------
    # Gesture handling
    # ------------------------------------------------------------------

    def _detect_gestures(self, frame: np.ndarray) -> List[Dict[str, object]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        gestures: List[Dict[str, object]] = []

        if not results.multi_hand_landmarks:
            return gestures

        for hand_landmarks in results.multi_hand_landmarks:
            score = self._classify_thumbsup(hand_landmarks)
            if score is None:
                continue
            gestures.append(
                {
                    "label": self.gesture_label,
                    "confidence": round(score, 4),
                    "command": "activate",
                }
            )

        return gestures

    def _classify_thumbsup(self, hand_landmarks) -> Optional[float]:
        landmarks = hand_landmarks.landmark

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Thumb extended upwards if tip higher (lower y) than MCP of index and IP of thumb
        thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < index_mcp.y

        # Other fingers should be curled (tips below their MCPs in image coordinates)
        index_folded = index_tip.y > landmarks[6].y
        middle_folded = middle_tip.y > landmarks[10].y
        ring_folded = ring_tip.y > landmarks[14].y
        pinky_folded = pinky_tip.y > landmarks[18].y

        if thumb_up and index_folded and middle_folded and ring_folded and pinky_folded:
            # Confidence heuristic based on distance between thumb tip and index MCP
            confidence = max(0.5, min(1.0, 1.0 - abs(thumb_tip.y - index_mcp.y)))
            return confidence
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release heavy resources such as MediaPipe sessions."""

        if self.mp_hands:
            self.mp_hands.close()


__all__ = ["FrameProcessor", "FrameDecodeError"]

