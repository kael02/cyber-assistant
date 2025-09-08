from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ConversionRequest(BaseModel):
    natural_language: str = Field(..., description="Natural language query to convert")
    context: str = Field("", description="Additional context for conversion")
    session_id: str = Field("default", description="Session ID for conversation tracking")


class CorrectionRequest(BaseModel):
    original_nl: str = Field(..., description="Original natural language input")
    original_query: str = Field(..., description="Original generated query")
    corrected_query: str = Field(..., description="Corrected query")
    feedback: str = Field("", description="User feedback about the correction")


class AnalysisRequest(BaseModel):
    analysis_type: str = Field("overall", description="Type of analysis (accuracy, formats, complexity, topics, improvements, overall)")
    time_period: str = Field("month", description="Time period for analysis (day, week, month, year)")


class SuggestionRequest(BaseModel):
    query_type: str = Field("general", description="Type of query (general, network, process, file, user, malware)")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Memory search query")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field("default", description="Session ID for conversation tracking")


class ProcessRequest(BaseModel):
    user_input: str = Field(..., description="User input to process through workflow")
    session_id: str = Field("default", description="Session ID for conversation tracking")

class BatchProcessRequest(BaseModel):
    """Request model for batch processing operations."""
    user_input: str = Field(..., description="User input to process through the complete workflow")
    session_id: Optional[str] = Field(None, description="Session identifier for this request")
    priority: Optional[int] = Field(1, description="Processing priority (1-10, higher = more priority)", ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for this request")

   