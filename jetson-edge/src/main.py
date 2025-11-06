"""
DETECH Jetson Edge - Main entry point
Handles video stream processing, object detection, and gesture recognition
"""

import asyncio
import cv2
import json
import time
from typing import Optional
from src.detectors.yolo_detector import YOLODetector
from src.gestures.gesture_handler import GestureHandler
from src.mqtt.mqtt_client import MQTTClient


class JetsonEdgeProcessor:
    """Main processor for Jetson edge device"""
    
    def __init__(self):
        self.yolo_detector: Optional[YOLODetector] = None
        self.gesture_handler: Optional[GestureHandler] = None
        self.mqtt_client: Optional[MQTTClient] = None
        self.running = False
    
    async def initialize(self):
        """Initialize detectors and MQTT client"""
        print("Initializing Jetson Edge Processor...")
        
        # Initialize YOLO11 detector
        self.yolo_detector = YOLODetector()
        await self.yolo_detector.load_model()
        
        # Initialize MediaPipe gesture handler
        self.gesture_handler = GestureHandler()
        await self.gesture_handler.initialize()
        
        # Initialize MQTT client
        self.mqtt_client = MQTTClient()
        await self.mqtt_client.connect()
        
        print("Jetson Edge Processor initialized")
    
    async def process_frame(self, frame: cv2.Mat) -> dict:
        """Process a single video frame"""
        detections = {
            "timestamp": time.time(),
            "objects": [],
            "gestures": [],
        }
        
        # Run YOLO11 object detection
        if self.yolo_detector:
            objects = await self.yolo_detector.detect(frame)
            detections["objects"] = objects
        
        # Run MediaPipe gesture recognition
        if self.gesture_handler:
            gestures = await self.gesture_handler.detect(frame)
            detections["gestures"] = gestures
        
        return detections
    
    async def process_stream(self, stream_source: str):
        """Process video stream from source"""
        self.running = True
        
        # Open video source (can be camera, file, or network stream)
        cap = cv2.VideoCapture(stream_source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open stream source: {stream_source}")
        
        print(f"Processing stream from: {stream_source}")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections = await self.process_frame(frame)
                
                # Publish detections via MQTT
                if self.mqtt_client and self.mqtt_client.is_connected():
                    await self.mqtt_client.publish_detection(detections)
                
                # Small delay to control processing rate
                await asyncio.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()
            print("Stream processing stopped")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.mqtt_client:
            await self.mqtt_client.disconnect()
        if self.yolo_detector:
            await self.yolo_detector.cleanup()
        if self.gesture_handler:
            await self.gesture_handler.cleanup()
        print("Jetson Edge Processor cleaned up")


async def main():
    """Main entry point"""
    processor = JetsonEdgeProcessor()
    
    try:
        await processor.initialize()
        
        # TODO: Get stream source from MQTT or configuration
        # For now, use default camera (0) or a test video
        stream_source = 0  # Change to video file path or network stream URL
        
        await processor.process_stream(stream_source)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
