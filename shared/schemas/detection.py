"""
Pydantic models for detections from Jetson edge device
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class ObjectDetection(BaseModel):
    """Object detection from YOLO11"""
    name: str = Field(description="Object class name")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence (0-1)")
    bbox: List[int] = Field(
        description="Bounding box [x, y, width, height]",
        min_length=4,
        max_length=4
    )


class GestureDetection(BaseModel):
    """Gesture detection from MediaPipe"""
    name: str = Field(description="Gesture name")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence (0-1)")
    landmarks: Optional[List[List[float]]] = Field(
        default=None,
        description="Hand landmarks [[x, y, z], ...]"
    )


class Detection(BaseModel):
    """Complete detection from Jetson edge device"""
    timestamp: float = Field(description="Unix timestamp of detection")
    objects: List[ObjectDetection] = Field(default_factory=list, description="Detected objects")
    gestures: List[GestureDetection] = Field(default_factory=list, description="Detected gestures")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1699123456.789,
                "objects": [
                    {
                        "name": "person",
                        "confidence": 0.95,
                        "bbox": [100, 200, 150, 300]
                    }
                ],
                "gestures": [
                    {
                        "name": "pointing",
                        "confidence": 0.87,
                        "landmarks": []
                    }
                ]
            }
        }
