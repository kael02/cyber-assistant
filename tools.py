from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import re
from datetime import datetime
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from config import logger


# Define structured output schemas
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


class CyberQueryTools:
    """Enhanced tools manager designed for workflow-only architecture with structured output."""

    def __init__(self, memory_tool, memory_system, field_store, field_context_k, llm, finetuned_llm, system_prompts):
        self.memory_tool = memory_tool
        self.memory_system = memory_system
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.llm = llm
        self.finetuned_llm = finetuned_llm
        self.system_prompts = system_prompts
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Caching for frequently accessed data
        self._field_cache = {}
        self._cache_lock = threading.Lock()
        
        # Create structured output models
        self._setup_structured_models()

    def _setup_structured_models(self):
        """Initialize structured output models."""
        try:
            # Create structured LLM for query conversion with Pydantic (preferred for validation)
            self.structured_query_llm = self.finetuned_llm.with_structured_output(
                QueryConversionResult,
                method="function_calling"  # Use function calling if available
            )
            
            # Create streaming version with TypedDict for potential streaming use
            self.streaming_query_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccessDict,
                method="function_calling"
            )
            
            logger.info("Structured output models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize structured models: {e}")
            # Fallback to non-structured approach
            self.structured_query_llm = self.finetuned_llm
            self.streaming_query_llm = self.finetuned_llm

    def _build_conversation_context(self, conversation_history: list, max_messages: int = 4) -> str:
        """Build contextual information from recent conversation history."""
        if not conversation_history:
            return ""
        
        try:
            # Take recent messages (exclude system messages)
            recent_messages = [
                msg for msg in conversation_history[-max_messages:] 
                if hasattr(msg, 'content') and msg.content
            ]
            
            if not recent_messages:
                return ""
            
            context_parts = []
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content[:200]}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Assistant: {msg.content[:200]}")
                elif hasattr(msg, 'type'):
                    if msg.type == 'human':
                        context_parts.append(f"User: {msg.content[:200]}")
                    elif msg.type == 'ai':
                        context_parts.append(f"Assistant: {msg.content[:200]}")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return ""

    def _extract_previous_query_context(self, conversation_history: list) -> Dict[str, Any]:
        """Extract context from previous queries to understand follow-up intent."""
        if not conversation_history:
            return {}
        
        context = {
            'has_previous_query': False,
            'previous_query': '',
            'previous_conditions': [],
            'query_type': '',
            'continuation_indicators': []
        }
        
        try:
            # Look for the most recent AI response containing a query
            for msg in reversed(conversation_history):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    
                    # Check if this contains a generated query
                    if '```' in content or any(keyword in content.lower() for keyword in ['source_ip:', 'destination_ip:', 'SELECT', 'WHERE']):
                        context['has_previous_query'] = True
                        context['previous_query'] = content
                        
                        # Extract conditions from the query
                        context['previous_conditions'] = self._extract_query_conditions(content)
                        break
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return context

    def _extract_query_conditions(self, query_text: str) -> List[str]:
        """Extract conditions from a previously generated query."""
        conditions = []
        
        try:
            # Common patterns in security queries
            patterns = [
                r'source_ip:\s*([^\s\)]+)',
                r'destination_ip:\s*([^\s\)]+)',
                r'port:\s*([^\s\)]+)',
                r'protocol:\s*([^\s\)]+)',
                r'user:\s*([^\s\)]+)',
                r'process:\s*([^\s\)]+)',
                r'file:\s*([^\s\)]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    conditions.append(match.strip('"\''))
            
        except Exception as e:
            logger.error(f"Error extracting conditions: {e}")
            
        return conditions

    def _detect_continuation_intent(self, user_input: str, conversation_history: list) -> Dict[str, Any]:
        """Detect if the user input is a continuation of a previous query."""
        continuation_info = {
            'is_continuation': False,
            'conjunction_type': 'AND',
            'new_conditions': [],
            'modification_type': 'add'
        }
        
        user_lower = user_input.lower().strip()
        
        # Strong continuation indicators
        continuation_patterns = [
            (r'^and\s+(.+)', 'AND', 'add'),
            (r'^also\s+(.+)', 'AND', 'add'),
            (r'^plus\s+(.+)', 'AND', 'add'),
            (r'^with\s+(.+)', 'AND', 'add'),
            (r'^including\s+(.+)', 'AND', 'add'),
            (r'^or\s+(.+)', 'OR', 'add'),
            (r'^but\s+not\s+(.+)', 'NOT', 'exclude'),
            (r'^exclude\s+(.+)', 'NOT', 'exclude'),
            (r'^except\s+(.+)', 'NOT', 'exclude')
        ]
        
        for pattern, conjunction, modification in continuation_patterns:
            match = re.match(pattern, user_lower)
            if match:
                continuation_info['is_continuation'] = True
                continuation_info['conjunction_type'] = conjunction
                continuation_info['modification_type'] = modification
                continuation_info['new_conditions'] = [match.group(1)]
                break
        
        # Check for implicit continuation
        if not continuation_info['is_continuation'] and conversation_history:
            previous_context = self._extract_previous_query_context(conversation_history)
            if (previous_context['has_previous_query'] and 
                self._has_similar_field_references(user_input, previous_context['previous_conditions'])):
                continuation_info['is_continuation'] = True
                continuation_info['new_conditions'] = [user_input]
        
        return continuation_info

    def _has_similar_field_references(self, current_input: str, previous_conditions: List[str]) -> bool:
        """Check if current input references similar fields as previous conditions."""
        current_lower = current_input.lower()
        
        field_patterns = {
            'ip': ['ip', 'address'],
            'port': ['port'],
            'user': ['user', 'username'],
            'process': ['process', 'executable'],
            'file': ['file', 'path'],
            'protocol': ['protocol', 'tcp', 'udp']
        }
        
        for condition in previous_conditions:
            for field_type, keywords in field_patterns.items():
                if any(keyword in condition.lower() for keyword in keywords):
                    if any(keyword in current_lower for keyword in keywords):
                        return True
        
        return False

    @lru_cache(maxsize=100)
    def _build_field_context_cached(self, nl_query: str, k: Optional[int] = None) -> str:
        """Cached version of field context building."""
        try:
            k = k or self.field_context_k
            hits = self.field_store.search_fields(nl_query, limit=k) or []
            if not hits:
                return ""
            
            lines = []
            for h in hits:
                alias = (h.get("alias") or "").strip()
                long_name = (h.get("long_name") or "")[:100].strip()
                dtype = (h.get("data_type") or "").strip()
                definition = (h.get("definition") or "")[:120].strip()
                
                lines.append(f"- {alias} | {dtype} | {long_name}")
                if len(lines) >= 5:
                    break
            
            return "Field context:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Field context error: {e}")
            return ""

    def _record_conversion_async(self, user_input: str, converted_query: str):
        """Async recording to avoid blocking."""
        def record():
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                record = f"Time: {current_time}\nInput: {user_input}\nQuery: {converted_query}"
                # Record in memory system
                logger.debug(f"Recording conversion: {user_input[:50]}...")
            except Exception as e:
                logger.error(f"Recording error: {e}")
        
        self.executor.submit(record)

    def convert_to_query_tool(
        self, 
        natural_language: str, 
        context: str = "", 
        conversation_history: list = None, 
        session_id: str = None,
        use_streaming: bool = False
    ) -> str:
        """Enhanced query conversion using structured output instead of prompt engineering."""
        logger.info(f"Converting with structured output: '{natural_language[:50]}...'")
        
        try:
            start_time = datetime.now()

            # Skip memory search for simple queries
            skip_memory = len(natural_language.split()) < 3
            
            if not skip_memory:
                field_context = self._build_field_context_cached(natural_language[:100])
            else:
                field_context = ""
        
            # Build conversation context
            conv_context = self._build_conversation_context(conversation_history or [])

            # Analyze for continuation intent
            previous_context = self._extract_previous_query_context(conversation_history or [])
            continuation_info = self._detect_continuation_intent(natural_language, conversation_history or [])

            # Enhanced system message for structured output
            system_content = f"""You are an expert security query translator with advanced conversational context awareness.

CORE RESPONSIBILITY: Convert natural language to structured query like graylog format OR identify out-of-scope requests.

RELEVANCE SCOPE:
✅ IN SCOPE: security logs, network analysis, malware detection, user authentication, file monitoring, 
process analysis, threat hunting, incident response, system monitoring, vulnerability scanning
❌ OUT OF SCOPE: weather, cooking, personal life, general knowledge, math problems, entertainment, 
non-security topics

CONVERSION GUIDELINES:
1. For relevant security queries: Return QueryConversionSuccess with proper structured query
2. For out-of-scope requests: Return QueryConversionError with OUT_OF_SCOPE error
3. Use available field context to ensure accurate field mapping
4. Handle conversational continuations by combining with previous query context
5. Calculate epoch timestamps when explicit time constraints are mentioned

CURRENT TIME: {int(datetime.now().timestamp())}

AVAILABLE FIELDS: {field_context if field_context else 'Use common security log fields'}
"""
            
            if conv_context:
                system_content += f"\n\nRECENT CONVERSATION:\n{conv_context}"

            # Build user message with continuation context
            user_message_parts = []
            
            # Add continuation analysis if detected
            if continuation_info['is_continuation'] and previous_context['has_previous_query']:
                user_message_parts.append(f"""CONTINUATION DETECTED:
Combining with previous query using {continuation_info['conjunction_type']} logic.
Previous Query: {previous_context['previous_query'][:200]}
Previous Conditions: {previous_context['previous_conditions']}
New conditions: {continuation_info['new_conditions']}""")
            
            if context:
                user_message_parts.append(f"ADDITIONAL CONTEXT: {context}")
            
            user_message_parts.append(f"CONVERT THIS: {natural_language}")
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content="\n\n".join(user_message_parts))
            ]

            # Use structured output model
            if use_streaming and hasattr(self.streaming_query_llm, 'stream'):
                # Use streaming version if requested and available
                result = self.streaming_query_llm.invoke(messages)
                response_content = self._format_structured_response(result, is_streaming=True)
            else:
                # Use standard structured output
                result = self.structured_query_llm.invoke(messages)
                response_content = self._format_structured_response(result)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Structured conversion completed in {duration:.2f}s")

            # Async recording for successful conversions
            if not self._is_error_response(response_content):
                self._record_conversion_async(natural_language, response_content)

            return response_content

        except Exception as e:
            logger.error(f"Structured conversion error: {str(e)}")
            return self._format_error_response("CONVERSION_FAILED", 
                                             "Conversion failed due to system error. Please try a simpler query format.")

    def _format_structured_response(self, result: Union[QueryConversionResult, dict], is_streaming: bool = False) -> str:
        """Format the structured output result into a user-friendly response."""
        try:
            if is_streaming:
                # Handle TypedDict result from streaming
                if isinstance(result, dict):
                    if 'error' in result:
                        return self._format_error_response(result['error'], result['message'], 
                                                         result.get('suggestions'))
                    else:
                        return result.get('query', 'No query generated')
            else:
                # Handle Pydantic result
                if hasattr(result, 'result'):
                    inner_result = result.result
                    if isinstance(inner_result, QueryConversionSuccess):
                        return inner_result
                    elif isinstance(inner_result, QueryConversionError):
                        return self._format_error_response(inner_result.error, inner_result.message, 
                                                         inner_result.suggestions)
                elif isinstance(result, dict):
                    # Fallback for dict-like results
                    return result.get('query', str(result))
                
            # Fallback - return string representation
            return str(result)
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {e}")
            return self._format_error_response("CONVERSION_FAILED", 
                                             "Failed to format query conversion result.")

    def _format_error_response(self, error_type: str, message: str, suggestions: Optional[List[str]] = None) -> str:
        """Format error responses consistently."""
        if error_type == "OUT_OF_SCOPE":
            response = f"{message}"
            if suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
            return response
        else:
            return f"Error: {message}"

    def _is_error_response(self, response: str) -> bool:
        """Check if response indicates an error condition."""
        error_indicators = ['Error:', 'OUT_OF_SCOPE:', 'I\'m specialized in', 'not relevant to security']
        return any(indicator in response for indicator in error_indicators)

    # Memory search tool with session awareness
    def memory_search_tool(self, query: str, session_id: str = None) -> str:
        """Enhanced memory search with session awareness."""
        logger.info(f"Memory search: '{query[:50]}...'")
        try:
            start_time = datetime.now()
            
            # If session_id provided, search session-specific first
            if session_id and hasattr(self.memory_system, 'search_session_memories'):
                session_results = self.memory_system.search_session_memories(
                    session_id, query, limit=3
                )
                if session_results:
                    result = self._format_session_search_results(session_results)
                else:
                    result = self.memory_tool.execute(query=query)
            else:
                result = self.memory_tool.execute(query=query)
            
            duration = (datetime.now() - start_time).total_seconds()
            if duration > 3.0:
                logger.warning(f"Slow memory search: {duration:.2f}s")
            
            if result:
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... [truncated]"
                return result_str
            else:
                return "No relevant memories found."
                
        except Exception as e:
            logger.error(f"Memory search error: {str(e)}")
            return "Memory search temporarily unavailable."

    def _format_session_search_results(self, results: List[Dict]) -> str:
        """Format session search results."""
        if not results:
            return "No relevant memories found in this session."
        
        formatted = "Found in this session:\n\n"
        for i, result in enumerate(results[:3], 1):
            timestamp = result.get('timestamp', 'Unknown time')
            content = result.get('content', str(result))[:150]
            formatted += f"{i}. {timestamp}: {content}...\n"
        
        return formatted

    # Rest of the methods remain the same...
    def analyze_patterns_tool(self, analysis_type: str, time_period: str = "week", conversation_history: list = None, session_id: str = None) -> str:
        """Enhanced pattern analysis with conversation context."""
        logger.info(f"Pattern analysis: {analysis_type}")
        
        # Use shorter time periods for faster results
        time_period = "week" if time_period == "month" else time_period
        
        try:
            # Build search query
            search_map = {
                "accuracy": f"conversions accuracy {time_period}",
                "formats": f"query formats {time_period}",
                "complexity": f"query complexity {time_period}",
                "topics": f"security topics {time_period}",
                "overall": f"query patterns {time_period}",
            }

            query = search_map.get(analysis_type, f"{analysis_type} patterns")
            
            # Use session-aware memory search if possible
            if session_id:
                memories = self.memory_search_tool(query, session_id=session_id)
            else:
                memories = self.memory_search_tool(query)

            # Build conversation context
            conv_context = self._build_conversation_context(conversation_history or [])

            # Enhanced analysis prompt with context
            analysis_prompt = f"""
            Analyze these query patterns ({analysis_type}):
            {memories[:800]}
            """
            
            if conv_context:
                analysis_prompt += f"\n\nCurrent conversation context:\n{conv_context[:400]}"
            
            analysis_prompt += "\n\nProvide: 1) Key trends 2) Common formats 3) Quick recommendations"

            messages = [
                SystemMessage(content="Provide concise pattern analysis considering the conversation context."),
                HumanMessage(content=analysis_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return f"Pattern analysis for {analysis_type}: Unable to complete analysis at this time."

    def get_suggestions_tool(self, query_type: str, conversation_history: list = None, session_id: str = None) -> str:
        """Enhanced suggestions with conversation context."""
        logger.info(f"Suggestions for: {query_type}")
        
        # Build conversation context for dynamic suggestions
        conv_context = self._build_conversation_context(conversation_history or [])
        
        # Pre-built suggestions for speed
        quick_suggestions = {
            "network": """Network Query Examples:
• "Show network connections from suspicious IPs"
• "Find processes with unusual network activity"
• "Display DNS queries to malicious domains"
• "Search for lateral movement patterns"
• "Identify data exfiltration attempts"

Best Practices: Use specific IP ranges, timeframes, and protocols.""",
            
            "process": """Process Query Examples:
• "Find processes running from temp directories"
• "Show processes with unsigned executables"
• "Identify processes with high privilege escalation"
• "Search for processes spawning unusual children"
• "Find processes with suspicious command lines"

Best Practices: Include parent-child relationships and execution paths.""",
            
            "file": """File Query Examples:
• "Show files created in system directories"
• "Find files with unusual extensions"
• "Identify recently modified critical files"
• "Search for files with suspicious hashes"
• "Display files accessed by unauthorized users"
"""
        }
        
        # If we have conversation context and it's a known type, enhance the suggestions
        if query_type in quick_suggestions and conv_context:
            try:
                enhancement_prompt = f"""
                Based on this conversation context:
                {conv_context[:300]}
                
                Enhance these suggestions for {query_type} queries with more specific, contextual examples.
                Keep it concise and practical.
                """
                
                messages = [
                    SystemMessage(content="Provide enhanced, contextual security query suggestions."),
                    HumanMessage(content=enhancement_prompt),
                ]
                
                enhanced_response = self.llm.invoke(messages)
                return f"{quick_suggestions[query_type]}\n\nContext-specific suggestions:\n{enhanced_response.content}"
                
            except Exception:
                # Fallback to pre-built suggestions if enhancement fails
                return quick_suggestions[query_type]
        
        elif query_type in quick_suggestions:
            return quick_suggestions[query_type]
        
        # Fallback to dynamic generation for unknown types
        try:
            suggestion_prompt = f"""
            Provide 3 example queries and best practices for {query_type} security analysis.
            Keep it concise and practical.
            """
            
            if conv_context:
                suggestion_prompt += f"\n\nConversation context:\n{conv_context[:300]}"

            messages = [
                SystemMessage(content="Provide practical security query suggestions."),
                HumanMessage(content=suggestion_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Suggestion error: {str(e)}")
            return f"Query suggestions for {query_type}: Try specific field names, timeframes, and conditions for better results."

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, feedback: str = "", session_id: str = None) -> str:
        """Record user corrections for learning."""
        logger.info(f"Recording correction: {original_nl[:50]}...")
        
        try:
            current_time = datetime.now().isoformat()
            correction_record = {
                "timestamp": current_time,
                "original_natural_language": original_nl,
                "original_query": original_query,
                "corrected_query": corrected_query,
                "feedback": feedback,
                "session_id": session_id
            }
            
            # Record in memory system with structured format
            correction_text = f"""
            CORRECTION RECORD:
            Time: {current_time}
            Original Input: {original_nl}
            Incorrect Query: {original_query}
            Corrected Query: {corrected_query}
            Feedback: {feedback}
            Session: {session_id}
            """
            
            # This should be automatically recorded by the memory system
            logger.info("Correction recorded successfully")
            
            return f"Correction recorded successfully. This will help improve future query conversions. Thank you for the feedback!"
            
        except Exception as e:
            logger.error(f"Correction recording error: {e}")
            return "Unable to record correction at this time. Please try again later."

    def get_conversion_summary(self, session_id: str = None) -> str:
        """Generate conversion summary with session awareness."""
        try:
            today = date.today().strftime("%Y-%m-%d")
            
            # Try session-specific summary first if session_id provided
            if session_id:
                conversions = self.memory_search_tool(f"conversions {today}", session_id=session_id)
            else:
                conversions = self.memory_search_tool(f"conversions {today}")

            if not conversions or "No relevant" in conversions:
                return "Today's Summary: No conversions yet. Start by converting your first security query!"

            # Quick summary without LLM call for better performance
            lines = conversions.split('\n')
            count = sum(1 for line in lines if 'Query:' in line or 'Convert' in line)
            
            session_note = f" (Session: {session_id})" if session_id else ""
            
            return f"""Today's Conversion Summary{session_note}:
• Queries converted: {count}
• Most recent: {lines[0][:100] if lines else 'None'}...
• Status: All systems operational

Use 'analyze patterns' for detailed insights."""

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return f"Summary unavailable: {str(e)[:50]}"

    def create_tools(self) -> List[Tool]:
        """Create LangChain tools for the workflow."""
        logger.debug("Creating workflow tools...")

        # These tools are designed to be called by the workflow, not directly by the assistant
        tools = [
            Tool(
                name="memory_search",
                description="Search past queries, patterns, and conversation history",
                func=lambda x: self.memory_search_tool(x)
            ),
            Tool(
                name="convert_to_query", 
                description="Convert natural language to structured security queries using structured output",
                func=lambda x: self.convert_to_query_tool(x)
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

    # Helper methods for workflow integration
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
            if tool_name == "memory_search":
                return isinstance(input_data, str) and len(input_data.strip()) > 0
            elif tool_name == "convert_to_query":
                return isinstance(input_data, str) and len(input_data.strip()) > 0
            elif tool_name == "analyze_patterns":
                return isinstance(input_data, str)
            elif tool_name == "get_suggestions":
                return isinstance(input_data, str)
            elif tool_name == "record_correction":
                return isinstance(input_data, (str, dict))
            elif tool_name == "get_conversion_summary":
                return True  # No specific validation needed
            
            return False
            
        except Exception as e:
            logger.error(f"Tool input validation error: {e}")
            return False

    # Additional helper methods for structured output
    def get_structured_conversion_with_confidence(
        self, 
        natural_language: str, 
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Get structured conversion result with confidence scoring."""
        try:
            # Use structured output to get detailed result
            result = self.structured_query_llm.invoke([
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
                    'confidence': conversion.confidence or 0.8,  # Default confidence
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

    def validate_structured_output_support(self) -> Dict[str, bool]:
        """Check which structured output features are supported by the current LLM."""
        capabilities = {
            'function_calling': False,
            'json_mode': False,
            'streaming': False,
            'pydantic_support': False
        }
        
        try:
            # Test function calling
            test_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccess,
                method="function_calling"
            )
            capabilities['function_calling'] = True
            capabilities['pydantic_support'] = True
        except:
            pass
            
        try:
            # Test JSON mode
            test_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccessDict,
                method="json_mode"
            )
            capabilities['json_mode'] = True
        except:
            pass
            
        try:
            # Test streaming
            if hasattr(self.streaming_query_llm, 'stream'):
                capabilities['streaming'] = True
        except:
            pass
            
        logger.info(f"Structured output capabilities: {capabilities}")
        return capabilities

    def get_query_examples_structured(self, query_type: str = "general") -> List[Dict[str, str]]:
        """Get structured examples for different query types."""
        examples = {
            "network": [
                {
                    "natural_language": "Show me all connections to suspicious IP 192.168.1.100",
                    "structured_query": 'source_ip:"192.168.1.100" OR destination_ip:"192.168.1.100"',
                    "description": "Network connection analysis"
                },
                {
                    "natural_language": "Find DNS queries to malicious domains in the last hour",
                    "structured_query": 'query_type:"DNS" AND (domain:*malicious* OR threat_intel:true) AND timestamp:[now-1h TO now]',
                    "description": "DNS threat analysis with time constraint"
                }
            ],
            "process": [
                {
                    "natural_language": "Show processes running from temp directories",
                    "structured_query": 'process_path:(*temp* OR *tmp* OR *appdata\\local\\temp*)',
                    "description": "Process location analysis"
                },
                {
                    "natural_language": "Find unsigned executables running as system",
                    "structured_query": 'process_signed:false AND process_user:*system*',
                    "description": "Process integrity and privilege analysis"
                }
            ],
            "file": [
                {
                    "natural_language": "Show files created in system directories today",
                    "structured_query": 'file_path:(*system32* OR *windows*) AND file_created:[now/d TO now] AND event_type:file_create',
                    "description": "System file creation monitoring"
                }
            ]
        }
        
        return examples.get(query_type, examples["network"])

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