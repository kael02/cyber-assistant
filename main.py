from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import api, memory, batch, admin
from services import CyberQueryAssistant
from config import logger
# Global assistant instance
assistant = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing Security Query Assistant...")
    app.state.assistant = CyberQueryAssistant()
    logger.info("✅ Assistant initialized successfully!")
    yield
    logger.info("👋 Shutting down Security Query Assistant...")


app = FastAPI(
    title="Security Query Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Include routers
app.include_router(api.router, prefix="/api", tags=["API"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(batch.router, prefix="/api/batch", tags=["Batch"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

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
            "process": "/api/process"
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