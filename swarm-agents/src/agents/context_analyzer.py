"""
Context Analyzer Agent - Analyzes context around detections
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
import asyncio
from src.prompts.context_analyzer_prompts import CONTEXT_ANALYZER_PROMPT


class ContextAnalyzerAgent:
    """LangChain agent for analyzing context around alerts"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.5,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXT_ANALYZER_PROMPT),
            ("human", "{alert}"),
        ])
    
    async def analyze(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context around alert
        
        Args:
            alert: Refined alert from AlertRefinerAgent
            
        Returns:
            Context analysis with additional insights
        """
        alert_text = self._format_alert(alert)
        
        chain = self.prompt | self.llm
        
        try:
            response = await chain.ainvoke({"alert": alert_text})
            
            context = {
                "confidence": 0.8,  # TODO: Extract from LLM response
                "insights": response.content,
                "recommendations": [],  # TODO: Extract from LLM response
                "similar_cases": [],  # TODO: Extract from historical data
            }
            
            return context
            
        except Exception as e:
            print(f"Error analyzing context: {e}")
            return {
                "confidence": 0.5,
                "insights": "Context analysis unavailable",
                "recommendations": [],
                "similar_cases": [],
            }
    
    def _format_alert(self, alert: Dict[str, Any]) -> str:
        """Format alert dict into text for LLM"""
        text = f"Alert ID: {alert.get('id', 'unknown')}\n"
        text += f"Type: {alert.get('type', 'unknown')}\n"
        text += f"Severity: {alert.get('severity', 'unknown')}\n"
        text += f"Description: {alert.get('description', 'N/A')}\n"
        
        if alert.get('detection'):
            text += f"\nDetection details:\n{alert['detection']}\n"
        
        return text
