from fastapi import APIRouter, Depends, HTTPException
from models import MemorySearchRequest, StandardResponse
from dependencies import get_assistant
from services import CyberQueryAssistant

router = APIRouter()

@router.post("/search", response_model=StandardResponse)
async def search_memory(
    request: MemorySearchRequest,
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Search through conversation memory."""
    try:
        results = assistant.memory_search(request.query)
        
        return StandardResponse(
            success=True,
            message=results,
            data={"query": request.query}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))