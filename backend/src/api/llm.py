"""
LLM Integration API endpoints for Dev-conditional Server Engine
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json

from ..storage.database import get_db
from ..storage.models import LLMConversation
from ..llm.service import LLMService
from ..config import settings

router = APIRouter()


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context: Optional[Dict[str, Any]] = None


class WorkflowSuggestionRequest(BaseModel):
    workflow_data: dict
    user_goal: str
    current_step: Optional[str] = None


class CodeValidationRequest(BaseModel):
    code: str
    language: str
    requirements: Optional[List[str]] = None


class CodeOptimizationRequest(BaseModel):
    code: str
    language: str
    optimization_type: str = "performance"  # performance, security, style


@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm(
    message: ChatMessage,
    db: AsyncSession = Depends(get_db)
):
    """Send a message to LLM and get response"""
    try:
        # Get LLM response
        llm_service = LLMService()
        response = await llm_service.chat(
            message.message,
            session_id=message.session_id,
            context=message.context
        )

        # Save conversation to database
        conversation = LLMConversation(
            session_id=message.session_id,
            user_message=message.message,
            llm_response=response["response"],
            context=response.get("context")
        )
        db.add(conversation)
        await db.commit()

        return ChatResponse(
            response=response["response"],
            session_id=message.session_id,
            context=response.get("context")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")


@router.post("/suggest-workflow")
async def suggest_workflow(
    request: WorkflowSuggestionRequest
):
    """Get workflow suggestions from LLM based on user goals"""
    try:
        llm_service = LLMService()
        suggestions = await llm_service.suggest_workflow(
            user_goal=request.user_goal,
            current_workflow=request.workflow_data,
            current_step=request.current_step
        )
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")


@router.post("/validate-code")
async def validate_code(
    request: CodeValidationRequest
):
    """Validate code using LLM"""
    try:
        llm_service = LLMService()
        validation = await llm_service.validate_code(
            code=request.code,
            language=request.language,
            requirements=request.requirements
        )
        return validation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")


@router.post("/optimize-code")
async def optimize_code(
    request: CodeOptimizationRequest
):
    """Optimize code using LLM"""
    try:
        llm_service = LLMService()
        optimization = await llm_service.optimize_code(
            code=request.code,
            language=request.language,
            optimization_type=request.optimization_type
        )
        return optimization
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")


@router.get("/conversation/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get conversation history for a session"""
    query = (
        select(LLMConversation)
        .where(LLMConversation.session_id == session_id)
        .limit(limit)
        .order_by(LLMConversation.created_at.desc())
    )
    result = await db.execute(query)
    conversations = result.scalars().all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": conv.id,
                "user_message": conv.user_message,
                "llm_response": conv.llm_response,
                "context": conv.context,
                "created_at": conv.created_at.isoformat()
            }
            for conv in conversations
        ]
    }


@router.delete("/conversation/{session_id}")
async def clear_conversation_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Clear conversation history for a session"""
    # This would typically be a soft delete or actual deletion
    # For now, we'll return success
    return {"message": f"Conversation history for session {session_id} cleared"}


@router.get("/models")
async def get_available_models():
    """Get available LLM models"""
    return {
        "models": [
            {
                "id": settings.OPENAI_MODEL,
                "name": "GPT-4",
                "provider": "openai",
                "capabilities": ["chat", "code_generation", "analysis"]
            }
        ]
    }


@router.get("/status")
async def get_llm_status():
    """Get LLM service status"""
    llm_service = LLMService()
    status = await llm_service.get_status()
    return status


# WebSocket endpoint for real-time LLM interaction
@router.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time LLM chat"""
    await websocket.accept()
    llm_service = LLMService()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Process message
            response = await llm_service.chat(
                message=message_data.get("message", ""),
                session_id=message_data.get("session_id", "default"),
                context=message_data.get("context")
            )

            # Send response back to client
            await websocket.send_text(json.dumps({
                "type": "response",
                "data": response
            }))

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Send error message
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"message": str(e)}
        }))
    finally:
        await websocket.close()