from fastapi import APIRouter, Depends, HTTPException
from models import ConversionRequest, ConversionResponse, ChatRequest, StandardResponse
from dependencies import get_assistant
from services import CyberQueryAssistant
router = APIRouter()


@router.post("/convert", response_model=ConversionResponse)
async def convert_query(
    request: ConversionRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Convert natural language to structured security query."""
    try:
        converted_query = assistant.convert_to_query(
            natural_language=request.natural_language,
            context=request.context
        )
        
        # Get memory context if available
        memory_context = None
        try:
            memory_context = assistant.memory_search(f"similar queries: {request.natural_language[:50]}")
        except:
            pass
        
        return ConversionResponse(
            success=not converted_query.startswith("Error:"),
            converted_query=converted_query,
            memory_context=memory_context,
            error=None if not converted_query.startswith("Error:") else converted_query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(
    request: ChatRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Chat with the assistant using memory context."""
    try:
        response = assistant.chat_with_memory(
            user_input=request.message,
            session_id=request.session_id
        )
        
        return StandardResponse(
            success=True,
            message=response,
            data={"session_id": request.session_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))