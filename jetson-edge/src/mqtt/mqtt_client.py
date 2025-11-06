"""
MQTT client for Jetson edge device
"""

import json
import paho.mqtt.client as mqtt
from typing import Optional
import asyncio
from threading import Thread


class MQTTClient:
    """MQTT client for publishing detections"""
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_detections: str = "detech/detections",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_detections = topic_detections
        self.client: Optional[mqtt.Client] = None
        self._connected = False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print(f"Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
            self._connected = True
        else:
            print(f"Failed to connect to MQTT broker: {rc}")
            self._connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        print("Disconnected from MQTT broker")
        self._connected = False
    
    async def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client(client_id="detech-jetson-edge")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Connect in a thread to avoid blocking
            def connect_thread():
                self.client.connect(self.broker_host, self.broker_port, 60)
                self.client.loop_start()
            
            thread = Thread(target=connect_thread)
            thread.start()
            thread.join(timeout=5)
            
            # Wait for connection
            await asyncio.sleep(1)
            
            if not self._connected:
                raise ConnectionError("Failed to connect to MQTT broker")
                
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            raise
    
    async def publish_detection(self, detection: dict):
        """Publish detection to MQTT broker"""
        if not self._connected or not self.client:
            print("MQTT client not connected")
            return
        
        try:
            payload = json.dumps(detection)
            result = self.client.publish(self.topic_detections, payload, qos=1)
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"Failed to publish detection: {result.rc}")
            else:
                print(f"Published detection: {detection.get('timestamp', 'unknown')}")
                
        except Exception as e:
            print(f"Error publishing detection: {e}")
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if MQTT client is connected"""
        return self._connected
