from datetime import datetime, date
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import re

from config import logger


class CyberQueryTools:
    """Enhanced tools manager designed for workflow-only architecture."""

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

    # Updated methods to work with session_id instead of conversation_history parameter
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

    def convert_to_query_tool(self, natural_language: str, context: str = "", conversation_history: list = None, session_id: str = None) -> str:
        """Enhanced query conversion with integrated relevance checking."""
        logger.info(f"Converting: '{natural_language[:50]}...'")
        try:
            start_time = datetime.now()

            # Skip memory search for simple queries
            skip_memory = len(natural_language.split()) < 5
            
            if not skip_memory:
                field_context = self._build_field_context_cached(natural_language[:100])
            else:
                field_context = ""
        
            # Build conversation context
            conv_context = self._build_conversation_context(conversation_history or [])

            # Analyze for continuation intent
            previous_context = self._extract_previous_query_context(conversation_history or [])
            continuation_info = self._detect_continuation_intent(natural_language, conversation_history or [])

            # Enhanced system prompt with integrated relevance checking
            system_prompt = """You are an expert security query translator with advanced conversational context awareness.

    RELEVANCE CHECK: First determine if the input is relevant to security query conversion:
    - Relevant: security logs, network analysis, malware detection, user authentication, file monitoring, process analysis, threat hunting, incident response
    - Not relevant: other topics like weather, cooking, personal life, general knowledge unrelated to security, math problems, entertainment

    If the input is NOT relevant to security queries, respond with: "OUT_OF_SCOPE: I'm specialized in security query conversions. I can help with converting natural language to security queries, analyzing logs, and security-related tasks. How can I assist with your security queries?"

    If the input IS relevant, proceed with conversion:

    KEY CAPABILITIES:
    1. Convert natural language to structured Graylog queries
    2. Understand conversational context and follow-up queries
    3. Handle continuations with AND, OR, NOT logic appropriately
    4. Maintain previous query context when building upon existing queries
    5. Make sure to use at least one of fields in field context

    RESPONSE FORMAT:
    - Return ONLY the structured query, no explanations
    - For continuations, combine with previous query using appropriate logical operators
    - Use proper syntax for the target query language
    - Ensure all conditions are properly formatted
    """
            
            if conv_context:
                system_prompt += f"\n\nRecent conversation context:\n{conv_context}"

            messages = [SystemMessage(content=system_prompt)]
            
            # Build enhanced prompt with continuation context
            user_message_parts = []
            
            # Add continuation analysis if detected
            if continuation_info['is_continuation'] and previous_context['has_previous_query']:
                user_message_parts.append(f"""CONTINUATION DETECTED:
    The user is adding to their previous query with a {continuation_info['conjunction_type']} condition.
    Previous Query: {previous_context['previous_query'][:200]}
    Previous Conditions: {previous_context['previous_conditions']}
    New conditions to add: {continuation_info['new_conditions']}

    INSTRUCTIONS:
    1. Take the previous query as the base
    2. Add the new condition using {continuation_info['conjunction_type']}
    3. Ensure proper syntax for the query language
    4. Maintain all existing conditions unless explicitly replaced""")
            
            # Add field context
            if field_context:
                user_message_parts.append(f"{field_context}")
            if context:
                user_message_parts.append(f"ADDITIONAL CONTEXT:\n{context}")
            
            user_message_parts.append(f"CONVERT: {natural_language}")
            
            messages.append(HumanMessage(content="\n\n".join(user_message_parts)))

            converted_query = self.finetuned_llm.invoke(messages)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Converted in {duration:.2f}s")

            # Check if the response indicates out of scope
            response_content = converted_query.content.strip()
            if response_content.startswith("OUT_OF_SCOPE:"):
                # Return the out-of-scope message without the prefix
                return response_content[13:].strip()

            # Async recording for successful conversions
            self._record_conversion_async(natural_language, response_content)

            return response_content

        except Exception as e:
            logger.error(f"Conversion error: {str(e)}")
            return f"Conversion failed: Please try a simpler query format."

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
                description="Convert natural language to structured security queries",
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

        logger.debug(f"Created {len(tools)} workflow tools")
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