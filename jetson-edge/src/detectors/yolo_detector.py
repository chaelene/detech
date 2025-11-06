"""
YOLO11 object detector for Jetson edge device
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import asyncio


class YOLODetector:
    """YOLO11 object detection handler"""
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO11 model file (will download if not exists)
        """
        self.model_path = model_path
        self.model: YOLO = None
        self._loaded = False
    
    async def load_model(self):
        """Load YOLO11 model"""
        try:
            print(f"Loading YOLO11 model: {self.model_path}")
            # YOLO will download the model if not present
            self.model = YOLO(self.model_path)
            self._loaded = True
            print("YOLO11 model loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLO11 model: {e}")
            raise
    
    async def detect(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLO11
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of detection dictionaries with:
            - name: Object class name
            - confidence: Detection confidence (0-1)
            - bbox: Bounding box [x, y, width, height]
        """
        if not self._loaded or not self.model:
            return []
        
        try:
            # Run YOLO inference
            # YOLO expects BGR format by default
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[cls_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Filter by confidence threshold
                    if confidence >= 0.5:  # 50% confidence threshold
                        detections.append({
                            "name": class_name,
                            "confidence": confidence,
                            "bbox": [x, y, width, height],
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self._loaded = False
