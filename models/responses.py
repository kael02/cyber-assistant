from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, TypedDict, Annotated, List
from datetime import datetime
from langchain_core.messages import BaseMessage


# Response models
class ConversionResponse(BaseModel):
    success: bool
    converted_query: str
    memory_context: Optional[str] = None
    error: Optional[str] = None
    is_query: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def last_write(prev, new):
    # reducer: pick the latest value
    return new if new is not None else prev

class AssistantState(TypedDict):
    """State definition for the Cyber Query Assistant workflow."""
    
    # Core message handling
    messages: List[BaseMessage]
    user_input: str
    
    # Operation mode and task management
    mode: str  # "converter", "analyzer", "pattern_recognition"
    current_task: str  # "query_conversion", "pattern_analysis", "suggestions", "memory_search", "general", "error"
    
    # Context and results
    memory_context: str
    conversion_result: str
    analysis_result: str
    error_message: str
    
    # Tool execution tracking
    tool_calls: List[Dict[str, Any]]
    
    # Session management
    session_id: str
    metadata: Dict[str, Any]


class QueryConversionRequest(TypedDict):
    """Request model for query conversion."""
    natural_language: str
    context: str
    session_id: str


class QueryConversionResponse(TypedDict):
    """Response model for query conversion."""
    converted_query: str
    field_context_used: bool
    similar_queries_found: bool
    conversion_time_seconds: float


class PatternAnalysisRequest(TypedDict):
    """Request model for pattern analysis."""
    analysis_type: str  # "accuracy", "formats", "complexity", "topics", "overall"
    time_period: str    # "day", "week", "month", "year"
    session_id: str


class PatternAnalysisResponse(TypedDict):
    """Response model for pattern analysis."""
    analysis_results: str
    patterns_found: int
    recommendations: List[str]
    analysis_time_seconds: float


class SuggestionRequest(TypedDict):
    """Request model for query suggestions."""
    query_type: str  # "network", "process", "file", "user", "malware", "general"
    session_id: str


class SuggestionResponse(TypedDict):
    """Response model for query suggestions."""
    suggestions: List[str]
    best_practices: List[str]
    example_queries: List[str]


class CorrectionRequest(TypedDict):
    """Request model for recording corrections."""
    original_natural_language: str
    original_generated_query: str
    corrected_query: str
    feedback: str
    session_id: str


class MemorySearchRequest(TypedDict):
    """Request model for memory search."""
    query: str
    session_id: str


class MemorySearchResponse(TypedDict):
    """Response model for memory search."""
    results: str
    results_found: bool
    search_time_seconds: float
