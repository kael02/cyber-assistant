from pydantic import BaseModel, Field
from typing import Optional, List, Union, TypedDict, Annotated


class QueryConversionSuccess(BaseModel):
    """Successful security query conversion result."""
    query: str = Field(description="The converted structured security query")
    from_timestamp: Optional[int] = Field(default=None, description="Start time as epoch timestamp")
    to_timestamp: Optional[int] = Field(default=None, description="End time as epoch timestamp")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1)")
    query_type: Optional[str] = Field(default=None, description="Type of security query (network, process, file, etc.)")

class QueryConversionError(BaseModel):
    """Query conversion error response."""
    error: str = Field(description="Error type: OUT_OF_SCOPE, INVALID_INPUT, or CONVERSION_FAILED")
    message: str = Field(description="Human-readable error message")
    suggestions: Optional[List[str]] = Field(default=None, description="Suggested alternatives")

class QueryConversionResult(BaseModel):
    """Union type for query conversion results."""
    result: Union[QueryConversionSuccess, QueryConversionError] = Field(description="Either success or error result")

# Alternative TypedDict versions for streaming support
class QueryConversionSuccessDict(TypedDict):
    """Successful security query conversion result (TypedDict version)."""
    query: Annotated[str, ..., "The converted structured security query"]
    from_timestamp: Annotated[Optional[int], None, "Start time as epoch timestamp"]
    to_timestamp: Annotated[Optional[int], None, "End time as epoch timestamp"]  
    confidence: Annotated[Optional[float], None, "Confidence score (0-1)"]
    query_type: Annotated[Optional[str], None, "Type of security query (network, process, file, etc.)"]

class QueryConversionErrorDict(TypedDict):
    """Query conversion error response (TypedDict version)."""
    error: Annotated[str, ..., "Error type: OUT_OF_SCOPE, INVALID_INPUT, or CONVERSION_FAILED"]
    message: Annotated[str, ..., "Human-readable error message"]
    suggestions: Annotated[Optional[List[str]], None, "Suggested alternatives"]