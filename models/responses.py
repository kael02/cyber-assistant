from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, TypedDict, Annotated, List
import operator



# Response models
class ConversionResponse(BaseModel):
    success: bool
    converted_query: str
    memory_context: Optional[str] = None
    error: Optional[str] = None


class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Define the state schema (same as original)
class AssistantState(TypedDict):
    messages: Annotated[List, operator.add]
    user_input: str
    mode: str
    current_task: str
    memory_context: str
    conversion_result: str
    analysis_result: str
    error_message: str
    tool_calls: List[Dict]
    session_id: str



