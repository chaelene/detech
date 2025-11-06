"""
MediaPipe gesture recognition handler
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any
import asyncio


class GestureHandler:
    """MediaPipe gesture recognition handler"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        self._initialized = False
        
        # Common gesture mappings
        self.gesture_names = {
            0: "fist",
            1: "open_palm",
            2: "thumbs_up",
            3: "thumbs_down",
            4: "pointing",
            5: "peace",
        }
    
    async def initialize(self):
        """Initialize MediaPipe hands solution"""
        try:
            print("Initializing MediaPipe gesture recognition...")
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._initialized = True
            print("MediaPipe gesture recognition initialized")
        except Exception as e:
            print(f"Failed to initialize MediaPipe: {e}")
            raise
    
    async def detect(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Detect gestures in frame using MediaPipe
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of gesture dictionaries with:
            - name: Gesture name
            - confidence: Detection confidence (0-1)
            - landmarks: Hand landmarks (if available)
        """
        if not self._initialized or not self.hands:
            return []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            gestures = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmark positions
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # Classify gesture (simplified - would use a proper classifier in production)
                    gesture_name, confidence = self._classify_gesture(landmarks)
                    
                    if gesture_name:
                        gestures.append({
                            "name": gesture_name,
                            "confidence": confidence,
                            "landmarks": landmarks,
                        })
            
            return gestures
            
        except Exception as e:
            print(f"Error during gesture detection: {e}")
            return []
    
    def _classify_gesture(self, landmarks: List[List[float]]) -> tuple[str, float]:
        """
        Classify gesture from landmarks
        
        This is a simplified classifier. In production, you would use
        a trained model or more sophisticated heuristics.
        
        Args:
            landmarks: List of [x, y, z] landmark positions
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if len(landmarks) < 21:  # MediaPipe hands has 21 landmarks
            return None, 0.0
        
        # Simple heuristic-based classification
        # TODO: Replace with proper trained model
        
        # Check if fingers are extended
        # Thumb (landmark 4)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        # Index finger (landmark 8)
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        # Middle finger (landmark 12)
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Ring finger (landmark 16)
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        # Pinky (landmark 20)
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Check if fingers are extended (tip y < pip y)
        index_extended = index_tip[1] < index_pip[1]
        middle_extended = middle_tip[1] < middle_pip[1]
        ring_extended = ring_tip[1] < ring_pip[1]
        pinky_extended = pinky_tip[1] < pinky_pip[1]
        thumb_extended = thumb_tip[0] > thumb_ip[0]  # Right hand
        
        # Classify based on extended fingers
        if all([index_extended, middle_extended, ring_extended, pinky_extended]):
            if thumb_extended:
                return "open_palm", 0.8
            else:
                return "pointing", 0.7
        elif index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            return "pointing", 0.9
        elif index_extended and middle_extended and not any([ring_extended, pinky_extended]):
            return "peace", 0.8
        elif not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "fist", 0.9
        
        return None, 0.0
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.hands:
            self.hands.close()
            self.hands = None
        self._initialized = False
