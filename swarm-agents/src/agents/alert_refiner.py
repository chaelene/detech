"""
Alert Refiner Agent - Refines raw detections into actionable alerts
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
import asyncio
from src.prompts.alert_refiner_prompts import ALERT_REFINER_PROMPT


class AlertRefinerAgent:
    """LangChain agent for refining detections into alerts"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,  # Lower temperature for more consistent analysis
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ALERT_REFINER_PROMPT),
            ("human", "{detection}"),
        ])
    
    async def refine(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine detection into alert
        
        Args:
            detection: Raw detection from Jetson edge
            
        Returns:
            Refined alert with severity, description, etc.
        """
        # Format detection for LLM
        detection_text = self._format_detection(detection)
        
        # Run LLM chain
        chain = self.prompt | self.llm
        
        try:
            response = await chain.ainvoke({"detection": detection_text})
            
            # Parse LLM response into alert structure
            alert = self._parse_response(response.content, detection)
            
            return alert
            
        except Exception as e:
            print(f"Error refining alert: {e}")
            # Return basic alert on error
            return self._create_basic_alert(detection)
    
    def _format_detection(self, detection: Dict[str, Any]) -> str:
        """Format detection dict into text for LLM"""
        text = f"Detection timestamp: {detection.get('timestamp', 'unknown')}\n"
        
        if detection.get('objects'):
            text += "Objects detected:\n"
            for obj in detection['objects']:
                text += f"  - {obj.get('name', 'unknown')} (confidence: {obj.get('confidence', 0):.2f})\n"
        
        if detection.get('gestures'):
            text += "Gestures detected:\n"
            for gesture in detection['gestures']:
                text += f"  - {gesture.get('name', 'unknown')} (confidence: {gesture.get('confidence', 0):.2f})\n"
        
        return text
    
    def _parse_response(self, response: str, original_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured alert"""
        # TODO: Use structured output from LLM (JSON mode)
        # For now, create a basic alert structure
        
        import uuid
        
        alert = {
            "id": str(uuid.uuid4()),
            "timestamp": original_detection.get('timestamp'),
            "type": "object_detection",  # Default, will be refined by LLM
            "severity": "medium",  # Default, will be refined by LLM
            "description": response[:200],  # Use LLM response as description
            "detection": original_detection,
            "refined_by": "alert_refiner",
        }
        
        # TODO: Extract structured fields from LLM response
        # This would use JSON mode or function calling
        
        return alert
    
    def _create_basic_alert(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic alert structure when LLM fails"""
        import uuid
        
        return {
            "id": str(uuid.uuid4()),
            "timestamp": detection.get('timestamp'),
            "type": "detection",
            "severity": "low",
            "description": "Detection received from edge device",
            "detection": detection,
            "refined_by": "fallback",
        }
