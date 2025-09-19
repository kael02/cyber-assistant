from fastapi import APIRouter, Depends, HTTPException
from models import ConversionRequest, ConversionResponse, ChatRequest, StandardResponse
from dependencies import get_assistant
from services import CyberQueryAssistant
router = APIRouter()
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio
from config import logger
from models.responses import ChatResponse
from datetime import datetime

@router.post("/convert", response_model=ConversionResponse)
async def convert_query(
    request: ConversionRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Convert natural language to structured security query with session memory."""
    try:
        # Use the session-aware workflow instead of direct tool call
        full_input = request.natural_language
            
        # This will use the full workflow including memory search and session state
        response = assistant.convert_to_query(
            natural_language=full_input,
            context=request.context,
            session_id=request.session_id
        )
        
        # Extract the converted query from the response
        # The response format from response_generation_node includes the converted query
        
        # Try to get memory context for this session
        memory_context = None
        try:
            memory_context = assistant.memory_search(
                f"session:{request.session_id} previous queries similar to: {request.natural_language[:50]}"
            )
        except Exception as e:
            logger.warning(f"Could not retrieve memory context: {e}")
        
        return ConversionResponse(
            success=response["success"],
            converted_query=response["response"],
            is_query=response["task_type"] == "conversion",
            memory_context=memory_context,
            error=None if response["success"] else response["response"]
        )
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert/stream-sse")
async def convert_query_stream_sse(
    request: ConversionRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Stream query using Server-Sent Events."""
    
    async def stream_sse() -> AsyncGenerator[str, None]:
        try:
            response = assistant.convert_to_query(
                natural_language=request.natural_language,
                context=request.context,
                session_id=request.session_id
            )
            
            # Stream word by word
            words = response["response"].split()
            accumulated = ""
            
            for word in words:
                if accumulated:
                    accumulated += " "
                accumulated += word
                
                yield f"data: {json.dumps({'query': accumulated})}\n\n"
                await asyncio.sleep(0.05)
                
            final_data = {
                'complete': True,
                'success': response['success'],
                'final_query': response['response']
            }
            yield f"data: {json.dumps(final_data)}\n\n"
      
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_memory(
    request: ChatRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Chat with full session memory and context awareness."""
    try:
        response = assistant.chat_with_memory(
            user_input=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Get conversation history for a specific session."""
    try:
        # Search for all conversations from this session
        history = assistant.memory_search(f"session:{session_id} conversations")
        
        return {
            "session_id": session_id,
            "history": history,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}/clear")
async def clear_session(
    session_id: str,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Clear session-specific cache and state."""
    try:
        # Clear the LangGraph memory for this session
        # Note: MemorySaver doesn't have a direct clear method, 
        # but you can implement one or recreate the saver
        
        # Clear cache entries for this session
        cache_keys_to_remove = [
            key for key in assistant._query_cache.keys() 
            if session_id in key
        ]
        for key in cache_keys_to_remove:
            del assistant._query_cache[key]
        
        return {
            "message": f"Session {session_id} cleared",
            "cleared_cache_entries": len(cache_keys_to_remove)
        }
        
    except Exception as e:
        logger.error(f"Session clear error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))