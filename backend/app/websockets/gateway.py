from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .connection_manager import manager
import json
import random
import asyncio

router = APIRouter()

@router.websocket("/stream/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        # Simulate initial handshake
        await manager.send_personal_message({"type": "info", "content": "Connected to Seismic Stream"}, user_id)
        
        while True:
            # Receive (Ping/Pong or Chat)
            # Use asyncio.wait or similar if we want to also push data proactively loop
            # For now, simple echo or trigger
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(data)
                
                if msg.get("type") == "chat":
                    await manager.broadcast({"sender": user_id, "content": msg.get("content"), "type": "chat"})
                
            except asyncio.TimeoutError:
                # Push simulated sensor data periodically
                # In real app, this would come from a background task or event bus
                if random.random() < 0.05: # 5% chance per 100ms
                    event = {
                        "type": "seismic_alert",
                        "severity": "high",
                        "location": "Sensor Node 1",
                        "timestamp": "Now"
                    }
                    await manager.send_personal_message(event, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        
