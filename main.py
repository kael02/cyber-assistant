from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import api, memory, batch, admin, document_ingestor
from services import CyberQueryAssistant
from config import logger, settings
from services.document_ingestor import DocumentIngestor

# Global assistant instance     
assistant = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing Security Query Assistant...")
    app.state.assistant = CyberQueryAssistant()
    logger.info("✅ Assistant initialized successfully!")
    app.state.ingestor = DocumentIngestor()
    logger.info("✅ Ingestor initialized successfully!")
    yield
    logger.info("👋 Shutting down Security Query Assistant...")
    

app = FastAPI(
    title="Security Query Assistant API",
    version="1.0.0",
    lifespan=lifespan
)



origins = settings.WHITELISTED_IPS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # only allow this origin
    allow_credentials=True,         # needed if you send cookies/auth
    allow_methods=["*"],            # allow all HTTP methods
    allow_headers=["*"],            # allow all request headers
)

# Include routers
app.include_router(api.router, prefix="/api", tags=["API"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(batch.router, prefix="/api/batch", tags=["Batch"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(document_ingestor.router, prefix="/api/document-ingestor", tags=["Document Ingester"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Security Query Assistant API",
        "version": "1.0.0",
        "description": "LangGraph-based Security Query Assistant with memory capabilities",
        "endpoints": {
            "convert": "/api/convert",
            "chat": "/api/chat",
            "analyze": "/api/analyze",
            "suggestions": "/api/suggestions",
            "memory_search": "/api/memory/search",
            "correction": "/api/correction",
            "summary": "/api/summary",
            "process": "/api/process",
            "document-ingestor": "/api/document-ingestor"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=True,  
        log_level="info"
    )