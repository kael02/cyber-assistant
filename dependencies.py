from fastapi import HTTPException
from fastapi import Request
from services import CyberQueryAssistant


def get_assistant(request: Request) -> CyberQueryAssistant:
    if not hasattr(request.app.state, 'assistant') or request.app.state.assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    return request.app.state.assistant