"""
DETECH Swarm Agents - Main orchestrator
LangChain-based collaborative analysis for refining alerts
"""

import asyncio
import json
from typing import List, Dict, Any
import redis
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agents.alert_refiner import AlertRefinerAgent
from src.agents.context_analyzer import ContextAnalyzerAgent
from src.tools.alert_tools import AlertTools


class SwarmOrchestrator:
    """Orchestrates multiple LangChain agents for collaborative analysis"""
    
    def __init__(self):
        self.redis_client: redis.Redis = None
        self.agents: List[Any] = []
        self.alert_tools = AlertTools()
        self._initialized = False
    
    async def initialize(self):
        """Initialize swarm agents and Redis connection"""
        print("Initializing Swarm Orchestrator...")
        
        # Connect to Redis for shared state
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Initialize agents
        self.alert_refiner = AlertRefinerAgent()
        self.context_analyzer = ContextAnalyzerAgent()
        
        self._initialized = True
        print("Swarm Orchestrator initialized")
    
    async def process_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process detection through swarm agents
        
        Args:
            detection: Detection from Jetson edge device
            
        Returns:
            Refined alert with swarm analysis
        """
        if not self._initialized:
            await self.initialize()
        
        # Store detection in Redis for agent access
        detection_id = f"detection:{detection.get('timestamp', 'unknown')}"
        await asyncio.to_thread(
            self.redis_client.setex,
            detection_id,
            3600,  # 1 hour TTL
            json.dumps(detection)
        )
        
        # Run alert refiner agent
        refined_alert = await self.alert_refiner.refine(detection)
        
        # Run context analyzer agent
        context_analysis = await self.context_analyzer.analyze(refined_alert)
        
        # Combine results
        final_alert = {
            **refined_alert,
            "context": context_analysis,
            "source": "swarm",
            "swarm_confidence": self._calculate_swarm_confidence(refined_alert, context_analysis),
        }
        
        # Publish alert to Redis for backend pickup
        alert_id = f"alert:{final_alert.get('id', 'unknown')}"
        await asyncio.to_thread(
            self.redis_client.setex,
            alert_id,
            3600,
            json.dumps(final_alert)
        )
        
        return final_alert
    
    def _calculate_swarm_confidence(self, refined: Dict, context: Dict) -> float:
        """Calculate confidence score from swarm analysis"""
        # Simple confidence calculation
        # TODO: Implement more sophisticated scoring
        base_confidence = refined.get('confidence', 0.5)
        context_confidence = context.get('confidence', 0.5)
        return (base_confidence + context_confidence) / 2
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
        self._initialized = False


async def main():
    """Main entry point for swarm service"""
    orchestrator = SwarmOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        # Subscribe to Redis for new detections
        pubsub = orchestrator.redis_client.pubsub()
        pubsub.subscribe('detech:detections')
        
        print("Swarm Orchestrator listening for detections...")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                detection = json.loads(message['data'])
                alert = await orchestrator.process_detection(detection)
                print(f"Generated alert: {alert.get('id', 'unknown')}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
