from typing import Union, Dict, Any, Optional, List
from models.tools import QueryConversionResult, QueryConversionSuccess, QueryConversionError, QueryConversionSuccessDict 
from config import logger

class StructuredOutputManager:
    """Manages structured LLM output for query conversion."""
    
    def __init__(self, llm):
        self.llm = llm
        self.structured_query_llm = None
        self.streaming_query_llm = None
        self._setup_structured_models()

    def _setup_structured_models(self):
        """Initialize structured output models."""
        try:
            
            self.structured_query_llm = self.llm.with_structured_output(
                QueryConversionResult,
                method="function_calling"
            )
            
            self.streaming_query_llm = self.llm.with_structured_output(
                QueryConversionSuccessDict,
                method="function_calling"
            )
            
            logger.info("Structured output models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize structured models: {e}")
            self.structured_query_llm = self.llm
            self.streaming_query_llm = self.llm

    def format_structured_response(self, result: Union[Any, dict], is_streaming: bool = False) -> Dict[str, Any]:
        """Format the structured output result into a user-friendly response."""
        try:
            if is_streaming:
                if isinstance(result, dict):
                    if 'error' in result:
                        return {
                            "response": self._format_error_response(result['error'], result['message'], 
                                                         result.get('suggestions')),
                            "metadata": {
                                "from_timestamp": None,
                                "to_timestamp": None,
                                "query_type": None,
                                "confidence": None
                            }
                        }
                    else:
                        return {
                            "response": result.get('query', 'No query generated'),
                            "metadata": {
                                "from_timestamp": result.get('from_timestamp'),
                                "to_timestamp": result.get('to_timestamp'),
                                "query_type": result.get('query_type'),
                                "confidence": result.get('confidence')
                            }
                        }
            else:
                if hasattr(result, 'result'):
                    inner_result = result.result
                    if isinstance(inner_result, QueryConversionSuccess):
                        return {
                            "response": inner_result.query,
                            "metadata": {
                                "from_timestamp": inner_result.from_timestamp,
                                "to_timestamp": inner_result.to_timestamp,
                                "query_type": inner_result.query_type,
                                "confidence": inner_result.confidence
                            }
                        }
                    elif isinstance(inner_result, QueryConversionError):
                        return {
                            "response": self._format_error_response(inner_result.error, inner_result.message, 
                                                         inner_result.suggestions),
                            "metadata": {
                                "from_timestamp": None,
                                "to_timestamp": None,
                                "query_type": None,
                                "confidence": None
                            }
                        }
                elif isinstance(result, dict):
                    return {
                        "response": result.get('query', str(result)),
                        "metadata": {
                            "from_timestamp": result.get('from_timestamp'),
                            "to_timestamp": result.get('to_timestamp'),
                            "query_type": result.get('query_type'),
                            "confidence": result.get('confidence')
                        }
                    }
                
            return {
                "response": str(result),
                "metadata": {
                    "from_timestamp": None,
                    "to_timestamp": None,
                    "query_type": None,
                    "confidence": None
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {e}")
            return {
                "response": self._format_error_response("CONVERSION_FAILED", 
                                             "Failed to format query conversion result."),
                "metadata": {
                    "from_timestamp": None,
                    "to_timestamp": None,
                    "query_type": None,
                    "confidence": None
                }
            }

    def _format_error_response(self, error_type: str, message: str, suggestions: Optional[List[str]] = None) -> str:
        """Format error responses consistently."""
        if error_type == "OUT_OF_SCOPE":
            response = f"{message}"
            if suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
            return response
        elif error_type == "NOT_SUPPORTED":
            response = f"{message}"
            if suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
            return response
        else:
            return f"Error: {message}"

    def validate_structured_output_support(self) -> Dict[str, bool]:
        """Check which structured output features are supported by the current LLM."""
        
        capabilities = {
            'function_calling': False,
            'json_mode': False,
            'streaming': False,
            'pydantic_support': False
        }
        
        try:
            test_llm = self.llm.with_structured_output(
                QueryConversionSuccess,
                method="function_calling"
            )
            capabilities['function_calling'] = True
            capabilities['pydantic_support'] = True
        except:
            pass
            
        try:
            test_llm = self.llm.with_structured_output(
                QueryConversionSuccessDict,
                method="json_mode"
            )
            capabilities['json_mode'] = True
        except:
            pass
            
        try:
            if hasattr(self.streaming_query_llm, 'stream'):
                capabilities['streaming'] = True
        except:
            pass
            
        logger.info(f"Structured output capabilities: {capabilities}")
        return capabilities