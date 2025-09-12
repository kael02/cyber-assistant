from fastapi import HTTPException
from fastapi import Request
from services import CyberQueryAssistant
from services.document_ingestor import DocumentIngestor


def get_assistant(request: Request) -> CyberQueryAssistant:
    if not hasattr(request.app.state, 'assistant') or request.app.state.assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    return request.app.state.assistant

def get_ingestor(request: Request) -> DocumentIngestor:
    if not hasattr(request.app.state, 'ingestor') or request.app.state.ingestor is None:
        raise HTTPException(status_code=503, detail="Ingestor not initialized")
    return request.app.state.ingestor