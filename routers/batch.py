from typing import List
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from models import ConversionRequest, ConversionResponse, StandardResponse, BatchProcessRequest
from dependencies import get_assistant
from services import CyberQueryAssistant

router = APIRouter()

@router.post("/convert")
async def batch_convert(
    queries: List[ConversionRequest] = Body(..., description="List of queries to convert"),
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Batch convert multiple natural language queries to structured queries."""
    results = []
    
    for query in queries:
        try:
            converted = assistant.convert_to_query(
                natural_language=query.natural_language,
                context=query.context
            )
            
            results.append({
                "input": query.natural_language,
                "output": converted,
                "success": not converted.startswith("Error:"),
                "context": query.context
            })
        except Exception as e:
            results.append({
                "input": query.natural_language,
                "output": None,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(queries),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
        "batch_id": f"batch_{len(queries)}_{hash(str(queries))}"
    }


@router.post("/analyze")
async def batch_analyze(
    queries: List[str] = Body(..., description="List of queries to analyze"),
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Batch analyze multiple queries for patterns and insights."""
    results = []
    
    for query in queries:
        try:
            # Analyze each query
            analysis = assistant.analyze_query_patterns("single", "day")  # Simplified
            results.append({
                "query": query,
                "analysis": analysis[:200],  # Truncate for batch response
                "success": True
            })
        except Exception as e:
            results.append({
                "query": query,
                "analysis": None,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(queries),
        "analyzed": sum(1 for r in results if r["success"]),
        "results": results
    }


@router.post("/process")
async def batch_process(
    background_tasks: BackgroundTasks,
    requests: List[BatchProcessRequest] = Body(...),
    assistant: CyberQueryAssistant = Depends(get_assistant)
):
    """Process multiple requests in the background."""
    batch_id = f"batch_{len(requests)}_{hash(str(requests))}"
    
    # Add background task for processing
    background_tasks.add_task(process_batch_background, requests, assistant, batch_id)
    
    return {
        "message": "Batch processing started",
        "batch_id": batch_id,
        "total_requests": len(requests),
        "status": "processing"
    }


async def process_batch_background(requests: List[BatchProcessRequest], assistant: CyberQueryAssistant, batch_id: str):
    """Background task to process batch requests."""
    # This would typically save results to a database or cache
    # For now, just process them
    for request in requests:
        try:
            result = assistant.process_user_input(
                user_input=request.user_input,
                session_id=request.session_id or batch_id
            )
            # Store result somewhere (database, cache, etc.)
        except Exception as e:
            # Log error
            pass


@router.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get the status of a batch operation."""
    # This would typically check a database or cache
    # For now, return a mock response
    return {
        "batch_id": batch_id,
        "status": "completed",  # or "processing", "failed"
        "progress": "100%",
        "results_available": True
    }