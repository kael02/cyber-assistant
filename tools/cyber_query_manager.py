from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import Tool
import threading
from config import logger
from models.tools import QueryConversionResult, QueryConversionSuccess, QueryConversionError
from .structured_output import StructuredOutputManager
from .contexts.conversation_context import ConversationContextManager
from .contexts.field_context import FieldContextManager
from .query_conversation import QueryConversionTool
from .memory_tools import MemoryTools
from .suggestion_tools import SuggestionTools
from .correction_tools import CorrectionTools
from langchain_core.messages import SystemMessage, HumanMessage

class CyberQueryManager:
    """Main manager class that orchestrates all cyber query tools."""
    
    def __init__(self, memory_tool, memory_system, field_store, field_context_k, 
                 llm, system_prompts):
        # Core dependencies
        self.memory_tool = memory_tool
        self.memory_system = memory_system
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.llm = llm
        self.system_prompts = system_prompts
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize modular components
        self._initialize_components()
        
        # Configuration
        self.example_context_k = 3
        self.combined_context_enabled = True
        self.example_relevance_threshold = 1.5

    def _initialize_components(self):
        """Initialize all modular components."""
        # Import modules
  
        
        # Initialize managers
        self.structured_output = StructuredOutputManager(self.llm)
        self.conversation_context = ConversationContextManager()
        self.field_context = FieldContextManager(self.field_store, self.field_context_k)
        
        # Initialize tools
        self.query_conversion = QueryConversionTool(
            self.structured_output, 
            self.conversation_context, 
            self.field_context
        )
        self.memory_tools = MemoryTools(self.memory_tool, self.memory_system, self.llm)
        self.suggestion_tools = SuggestionTools(self.llm)
        self.correction_tools = CorrectionTools(self.memory_system)

    # Main API methods
    def convert_to_query_tool(self, natural_language: str, context: str = "", 
                             conversation_history: list = None, session_id: str = None,
                             use_streaming: bool = False, include_examples: bool = True) -> Dict[str, Any]:
        """Convert natural language to structured query."""
        return self.query_conversion.convert_to_query(
            natural_language=natural_language,
            context=context,
            conversation_history=conversation_history,
            session_id=session_id,
            use_streaming=use_streaming,
            include_examples=include_examples
        )

    def memory_search_tool(self, query: str, session_id: str = None) -> str:
        """Search memory with session awareness."""
        return self.memory_tools.search_memory(query, session_id)

    def analyze_patterns_tool(self, analysis_type: str, time_period: str = "week", 
                             conversation_history: list = None, session_id: str = None) -> str:
        """Analyze patterns in query conversion history."""
        return self.memory_tools.analyze_patterns(
            analysis_type, time_period, conversation_history, session_id
        )

    def get_suggestions_tool(self, query_type: str, conversation_history: list = None, 
                            session_id: str = None) -> str:
        """Get query suggestions and examples."""
        return self.suggestion_tools.get_suggestions(query_type, conversation_history, session_id)

    def record_correction(self, original_nl: str, original_query: str, 
                         corrected_query: str, feedback: str = "", session_id: str = None) -> str:
        """Record user corrections for learning."""
        return self.correction_tools.record_correction(
            original_nl, original_query, corrected_query, feedback, session_id
        )

    def get_conversion_summary(self, session_id: str = None) -> str:
        """Generate conversion summary."""
        return self.correction_tools.get_conversion_summary(
            session_id, self.memory_tools.search_memory
        )

    # Advanced methods
    def get_structured_conversion_with_confidence(
        self, 
        natural_language: str, 
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Get structured conversion result with confidence scoring."""
        try:
            
            # Use structured output to get detailed result
            result = self.structured_output.structured_query_llm.invoke([
                SystemMessage(content="""Convert the security query and provide confidence scoring.
                
Return detailed conversion information including:
- The converted query
- Confidence score (0.0-1.0)  
- Query type classification
- Any relevant timestamps"""),
                HumanMessage(content=f"Convert: {natural_language}")
            ])
            
            if hasattr(result, 'result') and isinstance(result.result, QueryConversionSuccess):
                conversion = result.result
                return {
                    'success': True,
                    'query': conversion.query,
                    'confidence': conversion.confidence or 0.8,
                    'query_type': conversion.query_type or 'general',
                    'from_timestamp': conversion.from_timestamp,
                    'to_timestamp': conversion.to_timestamp,
                    'meets_threshold': (conversion.confidence or 0.8) >= confidence_threshold
                }
            else:
                return {
                    'success': False,
                    'error': 'Conversion failed or out of scope',
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"Structured conversion with confidence error: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }

    def batch_convert_queries(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Convert multiple queries using structured output for efficiency."""
        results = []
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"Processing batch query {i+1}/{len(queries)}")
                result = self.get_structured_conversion_with_confidence(query, **kwargs)
                results.append({
                    'index': i,
                    'original_query': query,
                    **result
                })
                
            except Exception as e:
                logger.error(f"Batch conversion error for query {i}: {e}")
                results.append({
                    'index': i,
                    'original_query': query,
                    'success': False,
                    'error': str(e),
                    'confidence': 0.0
                })
        
        return results

    def get_query_examples_structured(self, query_type: str = "general") -> List[Dict[str, str]]:
        """Get structured examples for different query types."""
        return self.suggestion_tools.get_query_examples_structured(query_type)

    def validate_structured_output_support(self) -> Dict[str, bool]:
        """Check which structured output features are supported."""
        return self.structured_output.validate_structured_output_support()

    def export_structured_config(self) -> Dict[str, Any]:
        """Export configuration for structured output setup."""
        capabilities = self.validate_structured_output_support()
        
        
        config = {
            'structured_output_enabled': any(capabilities.values()),
            'capabilities': capabilities,
            'schemas': {
                'success_schema': QueryConversionSuccess.schema(),
                'error_schema': QueryConversionError.schema(),
                'result_schema': QueryConversionResult.schema()
            },
            'field_context_enabled': bool(self.field_store),
            'memory_enabled': bool(self.memory_tool),
            'session_aware': hasattr(self.memory_system, 'search_session_memories')
        }
        
        logger.info("Structured output configuration exported")
        return config

    # Tool integration methods
    def create_tools(self) -> List[Tool]:
        """Create LangChain tools for the workflow."""
        logger.debug("Creating workflow tools...")

        tools = [
            Tool(
                name="memory_search",
                description="Search past queries, patterns, and conversation history",
                func=lambda x: self.memory_search_tool(x)
            ),
            Tool(
                name="convert_to_query", 
                description="Convert natural language to structured security queries using structured output",
                func=lambda x: str(self.convert_to_query_tool(x).get("response", ""))
            ),
            Tool(
                name="analyze_patterns",
                description="Analyze patterns in query conversion history", 
                func=lambda x: self.analyze_patterns_tool(x, "week")
            ),
            Tool(
                name="get_suggestions",
                description="Provide query suggestions and examples",
                func=lambda x: self.get_suggestions_tool(x)
            ),
            Tool(
                name="record_correction",
                description="Record user corrections and feedback",
                func=lambda x: "Please provide correction details in the format: original_query -> corrected_query"
            ),
            Tool(
                name="get_summary",
                description="Generate conversion summary",
                func=lambda x: self.get_conversion_summary()
            )
        ]

        logger.debug(f"Created {len(tools)} workflow tools with structured output support")
        return tools

    def get_tool_by_name(self, tool_name: str) -> Optional[callable]:
        """Get a specific tool function by name for workflow use."""
        tool_map = {
            "memory_search": self.memory_search_tool,
            "convert_to_query": self.convert_to_query_tool,
            "analyze_patterns": self.analyze_patterns_tool,
            "get_suggestions": self.get_suggestions_tool,
            "record_correction": self.record_correction,
            "get_conversion_summary": self.get_conversion_summary
        }
        
        return tool_map.get(tool_name)

    def validate_tool_input(self, tool_name: str, input_data: Any) -> bool:
        """Validate input for a specific tool."""
        try:
            validation_map = {
                "memory_search": lambda x: isinstance(x, str) and len(x.strip()) > 0,
                "convert_to_query": lambda x: isinstance(x, str) and len(x.strip()) > 0,
                "analyze_patterns": lambda x: isinstance(x, str),
                "get_suggestions": lambda x: isinstance(x, str),
                "record_correction": lambda x: isinstance(x, (str, dict)),
                "get_conversion_summary": lambda x: True
            }
            
            validator = validation_map.get(tool_name)
            return validator(input_data) if validator else False
            
        except Exception as e:
            logger.error(f"Tool input validation error: {e}")
            return False