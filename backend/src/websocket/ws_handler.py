"""
WebSocket handlers for real-time communication in Dev-conditional Server Engine
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> session_id

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str = None):
        """Connect a new WebSocket client"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = []

        self.active_connections[session_id].append(websocket)

        if user_id:
            self.user_connections[user_id] = session_id

        logger.info(f"New connection: session_id={session_id}, user_id={user_id}")

    def disconnect(self, websocket: WebSocket, session_id: str, user_id: str = None):
        """Disconnect a WebSocket client"""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

            # Remove session if no more connections
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]

        logger.info(f"Connection closed: session_id={session_id}, user_id={user_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket client"""
        await websocket.send_text(message)

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all clients in a session"""
        if session_id in self.active_connections:
            disconnected_clients = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected_clients.append(connection)

            # Clean up disconnected clients
            for client in disconnected_clients:
                self.active_connections[session_id].remove(client)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        for session_id in list(self.active_connections.keys()):
            await self.broadcast_to_session(session_id, message)

    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user"""
        if user_id in self.user_connections:
            session_id = self.user_connections[user_id]
            await self.broadcast_to_session(session_id, message)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str, user_id: str = None):
    """WebSocket endpoint for real-time LLM chat"""
    await manager.connect(websocket, session_id, user_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle different message types
            message_type = message_data.get("type", "chat")

            if message_type == "chat":
                # Process chat message
                await handle_chat_message(session_id, message_data)
            elif message_type == "workflow_update":
                # Handle workflow updates
                await handle_workflow_update(session_id, message_data)
            elif message_type == "code_generation":
                # Handle code generation updates
                await handle_code_generation_update(session_id, message_data)
            elif message_type == "ping":
                # Handle ping for connection health
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, session_id, user_id)


@router.websocket("/workflow/{workflow_id}")
async def websocket_workflow_endpoint(websocket: WebSocket, workflow_id: str, user_id: str = None):
    """WebSocket endpoint for workflow execution updates"""
    session_id = f"workflow_{workflow_id}"
    await manager.connect(websocket, session_id, user_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle workflow-specific messages
            message_type = message_data.get("type", "status")

            if message_type == "start_execution":
                await handle_workflow_execution_start(workflow_id, message_data)
            elif message_type == "stop_execution":
                await handle_workflow_execution_stop(workflow_id, message_data)
            elif message_type == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id, user_id)
    except Exception as e:
        logger.error(f"Workflow WebSocket error: {str(e)}")
        manager.disconnect(websocket, session_id, user_id)


async def handle_chat_message(session_id: str, message_data: dict):
    """Handle chat messages and integrate with LLM service"""
    try:
        from ..llm.service import LLMService

        llm_service = LLMService()
        response = await llm_service.chat(
            message=message_data.get("message", ""),
            session_id=session_id,
            context=message_data.get("context")
        )

        # Send response back to session
        await manager.broadcast_to_session(session_id, {
            "type": "chat_response",
            "data": response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        await manager.broadcast_to_session(session_id, {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })


async def handle_workflow_update(session_id: str, message_data: dict):
    """Handle workflow updates"""
    # Broadcast workflow update to session
    await manager.broadcast_to_session(session_id, {
        "type": "workflow_updated",
        "data": message_data.get("data"),
        "timestamp": datetime.now().isoformat()
    })


async def handle_code_generation_update(session_id: str, message_data: dict):
    """Handle code generation updates"""
    # Broadcast code generation status
    await manager.broadcast_to_session(session_id, {
        "type": "code_generation_update",
        "data": message_data.get("data"),
        "timestamp": datetime.now().isoformat()
    })


async def handle_workflow_execution_start(workflow_id: str, message_data: dict):
    """Handle workflow execution start"""
    session_id = f"workflow_{workflow_id}"
    await manager.broadcast_to_session(session_id, {
        "type": "execution_started",
        "workflow_id": workflow_id,
        "timestamp": datetime.now().isoformat()
    })


async def handle_workflow_execution_stop(workflow_id: str, message_data: dict):
    """Handle workflow execution stop"""
    session_id = f"workflow_{workflow_id}"
    await manager.broadcast_to_session(session_id, {
        "type": "execution_stopped",
        "workflow_id": workflow_id,
        "timestamp": datetime.now().isoformat()
    })


# Utility functions for external modules to use WebSocket communication
async def notify_workflow_update(workflow_id: str, update_data: dict):
    """Notify clients about workflow updates"""
    session_id = f"workflow_{workflow_id}"
    await manager.broadcast_to_session(session_id, {
        "type": "workflow_update",
        "data": update_data,
        "timestamp": datetime.now().isoformat()
    })


async def notify_code_generation_progress(project_id: str, progress_data: dict):
    """Notify clients about code generation progress"""
    await manager.broadcast_to_all({
        "type": "code_generation_progress",
        "project_id": project_id,
        "data": progress_data,
        "timestamp": datetime.now().isoformat()
    })


async def notify_llm_response(session_id: str, response_data: dict):
    """Notify clients about LLM responses"""
    await manager.broadcast_to_session(session_id, {
        "type": "llm_response",
        "data": response_data,
        "timestamp": datetime.now().isoformat()
    })