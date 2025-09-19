from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import re

from config import logger
from models import AssistantState


class CyberQueryWorkflow:
    """Simplified workflow manager with integrated relevance checking."""

    def __init__(self, tools_manager, memory_system, system_prompts, llm):
        self.tools_manager = tools_manager
        self.memory_system = memory_system
        self.system_prompts = system_prompts
        self.llm = llm
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def create_graph(self, memory_saver) -> StateGraph:
        """Create the simplified workflow."""
        logger.debug("Building simplified workflow...")

        workflow = StateGraph(AssistantState)
        
        # Simplified workflow nodes
        nodes = [
            ("load_session_context", self.load_session_context_node),
            ("classify_intent", self.classify_intent_node),
            ("handle_conversion", self.handle_conversion_node),
            ("handle_memory_search", self.handle_memory_search_node),
            ("handle_analysis", self.handle_analysis_node),
            ("handle_suggestions", self.handle_suggestions_node),
            ("handle_session_history", self.handle_session_history_node),
            ("handle_correction", self.handle_correction_node),
            ("response_generation", self.response_generation_node),
            ("error_handling", self.error_handling_node)
        ]

        for node_name, node_func in nodes:
            workflow.add_node(node_name, node_func)

        # Simplified workflow routing
        workflow.add_edge(START, "load_session_context")
        workflow.add_edge("load_session_context", "classify_intent")

        # Route based on classified intent (no separate relevance checking)
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "conversion": "handle_conversion",
                "memory_search": "handle_memory_search",
                "analysis": "handle_analysis",
                "suggestions": "handle_suggestions",
                "session_history": "handle_session_history",
                "correction": "handle_correction",
                "error": "error_handling"
            }
        )
        
        # All task handlers route to response generation
        task_handlers = [
            "handle_conversion", "handle_memory_search", "handle_analysis",
            "handle_suggestions", "handle_session_history", "handle_correction"
        ]
        
        for handler in task_handlers:
            workflow.add_conditional_edges(
                handler,
                self.route_after_task,
                {
                    "success": "response_generation",
                    "error": "error_handling"
                }
            )

        workflow.add_edge("response_generation", END)
        workflow.add_edge("error_handling", END)

        compiled_graph = workflow.compile(checkpointer=memory_saver)
        logger.debug("Simplified workflow compiled successfully")
        return compiled_graph

    def load_session_context_node(self, state: "AssistantState") -> "AssistantState":
        """Load session context and conversation history."""
        session_id = state.get("session_id", "default")
        
        try:
            # Load conversation history for this session
            conversation_history = self.memory_system.get_session_context(
                session_id, max_messages=10
            )
            
            logger.info(f"conversation_history: {conversation_history}")
            
            # Merge with existing messages (user input is already in messages)
            state["messages"] = conversation_history + state["messages"]
            
            logger.debug(f"Loaded {len(conversation_history)} messages for session {session_id}")
            
        except Exception as e:
            logger.error(f"Session context loading error: {e}")
            # Continue without session context
            
        return state

    def classify_intent_node(self, state: "AssistantState") -> "AssistantState":
        """Simplified intent classification without separate relevance checking."""
        user_input = state["user_input"]
        session_id = state.get("session_id", "default")
        conversation_history = state.get("messages", [])
        
        logger.info(f"Classifying intent for session {session_id}")
        
        try:
            # Get conversation context for better classification
            recent_context = self._get_recent_context(conversation_history)
            
            # Classification with context (no separate relevance check)
            intent = self._classify_with_context(user_input, recent_context)
            state["current_task"] = intent
            
            logger.info(f"Classified as: {intent}")
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            state["current_task"] = "error"
            state["error_message"] = f"Intent classification failed: {e}"
            
        return state

    def _get_recent_context(self, conversation_history: List[Any]) -> str:
        """Extract recent conversation context for classification."""
        if not conversation_history:
            return ""
        
        try:
            # Get last few messages
            recent_messages = conversation_history[-4:]
            context_parts = []
            
            for msg in recent_messages:
                if hasattr(msg, 'content') and msg.content:
                    msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    content = str(msg.content)[:150]  # Truncate for efficiency
                    context_parts.append(f"{msg_type}: {content}")
            
            return "\n".join(context_parts)
        except Exception:
            return ""

    def _classify_with_context(self, user_input: str, context: str) -> str:
        """Classify intent considering conversation context."""
        input_lower = user_input.lower()
        
        # Strong intent indicators
        strong_patterns = {
            "memory_search": [
                r"remember.*when", r"recall.*previous", r"search.*memories",
                r"what.*did.*we.*discuss", r"find.*in.*past"
            ],
            "analysis": [
                r"analyze.*patterns", r"show.*trends", r"pattern.*analysis",
                r"accuracy.*report", r"conversion.*statistics"
            ],
            "suggestions": [
                r"suggest.*queries", r"give.*examples", r"how.*to.*query",
                r"query.*suggestions", r"help.*with.*queries"
            ],
            "session_history": [
                r"show.*history", r"conversation.*history", r"what.*did.*we.*talk",
                r"previous.*messages", r"our.*conversation"
            ],
            "correction": [
                r"that.*was.*wrong", r"correct.*query", r"should.*be",
                r"fix.*the.*query", r"record.*correction"
            ]
        }
        
        # Check for strong patterns first
        for intent, patterns in strong_patterns.items():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                return intent
        
        # Context-based classification for ambiguous inputs
        if context:
            context_lower = context.lower()
            
            # If recent context was about conversion, and input looks like a follow-up
            if ("convert" in context_lower or "query" in context_lower) and len(user_input.split()) > 2:
                # Check for continuation indicators
                if any(word in input_lower for word in ["and", "also", "plus", "with", "or", "but not"]):
                    return "conversion"
        
        # Fallback to simple keyword matching
        keyword_map = {
            "memory_search": ["remember", "recall", "past", "previous", "memory"],
            "analysis": ["analyze", "pattern", "trends", "accuracy", "statistics"],
            "suggestions": ["suggest", "example", "help", "how to", "guide"],
            "session_history": ["history", "conversation", "messages", "talk"],
            "correction": ["wrong", "correct", "fix", "should be", "mistake"]
        }
        
        for intent, keywords in keyword_map.items():
            if any(keyword in input_lower for keyword in keywords):
                return intent
        
        # Default to conversion - let the conversion tool handle relevance checking
        return "conversion"

    def handle_conversion_node(self, state: "AssistantState") -> "AssistantState":
        """Handle query conversion with integrated relevance checking."""
        user_input = state["user_input"]
        session_id = state.get("session_id", "default")
        conversation_history = state.get("messages", [])
        
        logger.info("Handling query conversion...")
        
        try:
            # Use tools manager with conversation history - it now handles relevance checking internally
            result = self.tools_manager.convert_to_query_tool(
                natural_language=user_input,
                context="",
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            state["conversion_result"] = result
            
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            state["error_message"] = f"Conversion failed: {e}"
            state["current_task"] = "error"
            
        return state

    def handle_memory_search_node(self, state: "AssistantState") -> "AssistantState":
        """Handle memory search operations."""
        user_input = state["user_input"]
        session_id = state.get("session_id", "default")
        
        logger.info("Handling memory search...")
        
        try:
            # Extract search query from user input
            search_query = self._extract_search_query(user_input)
            
            # Search session-specific memories first
            session_results = self.memory_system.search_session_memories(
                session_id, search_query, limit=5
            )
            
            if session_results:
                state["memory_context"] = self._format_session_memories(session_results)
            else:
                # Fall back to global memory search
                global_results = self.tools_manager.memory_search_tool(search_query, session_id=session_id)
                state["memory_context"] = global_results
                
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            state["error_message"] = f"Memory search failed: {e}"
            state["current_task"] = "error"
            
        return state

    def handle_analysis_node(self, state: "AssistantState") -> "AssistantState":
        """Handle pattern analysis operations."""
        user_input = state["user_input"]
        conversation_history = state.get("messages", [])
        session_id = state.get("session_id", "default")
        
        logger.info("Handling pattern analysis...")
        
        try:
            analysis_type = self._extract_analysis_type(user_input)
            time_period = self._extract_time_period(user_input)
            
            result = self.tools_manager.analyze_patterns_tool(
                analysis_type=analysis_type,
                time_period=time_period,
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            state["analysis_result"] = result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            state["error_message"] = f"Analysis failed: {e}"
            state["current_task"] = "error"
            
        return state

    def handle_suggestions_node(self, state: "AssistantState") -> "AssistantState":
        """Handle query suggestions."""
        user_input = state["user_input"]
        conversation_history = state.get("messages", [])
        session_id = state.get("session_id", "default")
        
        logger.info("Handling query suggestions...")
        
        try:
            query_type = self._extract_query_type(user_input)
            
            result = self.tools_manager.get_suggestions_tool(
                query_type=query_type,
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            state["analysis_result"] = result  # Using analysis_result for suggestions
            
        except Exception as e:
            logger.error(f"Suggestions error: {e}")
            state["error_message"] = f"Suggestions failed: {e}"
            state["current_task"] = "error"
            
        return state

    def handle_session_history_node(self, state: "AssistantState") -> "AssistantState":
        """Handle session history requests."""
        session_id = state.get("session_id", "default")
        
        logger.info("Handling session history...")
        
        try:
            # Get session summary and conversations
            session_summary = self.memory_system.get_session_summary(session_id) if hasattr(self.memory_system, 'get_session_summary') else {}
            session_conversations = self.memory_system.get_session_context(session_id, max_messages=20)
            
            state["session_summary"] = session_summary
            state["session_conversations"] = session_conversations
            
        except Exception as e:
            logger.error(f"Session history error: {e}")
            state["error_message"] = f"Session history failed: {e}"
            state["current_task"] = "error"
            
        return state

    def handle_correction_node(self, state: "AssistantState") -> "AssistantState":
        """Handle correction recording."""
        user_input = state["user_input"]
        session_id = state.get("session_id", "default")
        
        logger.info("Handling correction...")
        
        try:
            # Extract correction details from input
            correction_info = self._extract_correction_info(user_input)
            
            if correction_info:
                result = self.tools_manager.record_correction(
                    original_nl=correction_info.get("original_nl", ""),
                    original_query=correction_info.get("original_query", ""),
                    corrected_query=correction_info.get("corrected_query", ""),
                    feedback=correction_info.get("feedback", ""),
                    session_id=session_id
                )
                state["analysis_result"] = result
            else:
                state["analysis_result"] = "I need more details to record the correction. Please provide the original query and the corrected version."
            
        except Exception as e:
            logger.error(f"Correction error: {e}")
            state["error_message"] = f"Correction failed: {e}"
            state["current_task"] = "error"
            
        return state

    def response_generation_node(self, state: "AssistantState") -> "AssistantState":
        """Generate final response based on task results."""
        logger.info("Generating final response...")
        
        try:
            task = state["current_task"]
            user_input = state["user_input"]
            session_id = state.get("session_id", "default")
            
            # Generate response based on task type and results
            if task == "conversion" and state.get("conversion_result"):
                response = self._format_conversion_response(
                    user_input, state["conversion_result"]
                )
                
            elif task == "memory_search" and state.get("memory_context"):
                response = f"Memory Search Results:\n\n{state['memory_context']}"
                
            elif task == "session_history":
                response = self._format_session_history_response(state)
                
            elif task in ["analysis", "suggestions", "correction"] and state.get("analysis_result"):
                response = state['analysis_result']
                
            else:
                # Fallback response
                response = self._generate_fallback_response(user_input, task)

            # Add response to message history
            state["messages"].append(AIMessage(content=response))
            
            # Record interaction in session memory
            self.memory_system.record_session_interaction(
                session_id, user_input, response
            )
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            state["error_message"] = f"Response generation failed: {e}"
            state["current_task"] = "error"
            
        return state

    def error_handling_node(self, state: "AssistantState") -> "AssistantState":
        """Handle errors gracefully."""
        error_msg = state.get('error_message', 'Unknown error occurred')
        logger.warning(f"Error handling: {error_msg}")
        
        # Generate user-friendly error response
        response = self._generate_error_response(error_msg, state["current_task"])
        state["messages"].append(AIMessage(content=response))
        
        return state

    # Routing functions
    def route_by_intent(self, state: "AssistantState") -> str:
        """Route to appropriate task handler based on classified intent."""
        task = state.get("current_task", "conversion")
        
        # Map tasks to handlers
        task_routes = {
            "conversion": "conversion",
            "memory_search": "memory_search", 
            "analysis": "analysis",
            "suggestions": "suggestions",
            "session_history": "session_history",
            "correction": "correction",
            "error": "error"
        }
        
        return task_routes.get(task, "conversion")  # Default to conversion

    def route_after_task(self, state: "AssistantState") -> str:
        """Route after task completion."""
        return "error" if state.get("error_message") else "success"

    # Helper methods for data extraction and formatting
    def _extract_search_query(self, user_input: str) -> str:
        """Extract search query from user input."""
        # Remove common prefixes
        query = user_input.lower()
        prefixes = ["search for", "find", "remember", "recall", "look for", "search memories for"]
        
        for prefix in prefixes:
            if query.startswith(prefix):
                return user_input[len(prefix):].strip()
        
        return user_input

    def _extract_analysis_type(self, user_input: str) -> str:
        """Extract analysis type from user input."""
        input_lower = user_input.lower()
        
        type_map = {
            "accuracy": ["accuracy", "correct", "performance"],
            "formats": ["format", "structure", "syntax"],
            "complexity": ["complex", "difficulty", "advanced"],
            "topics": ["topic", "subject", "theme", "category"],
        }
        
        for analysis_type, keywords in type_map.items():
            if any(keyword in input_lower for keyword in keywords):
                return analysis_type
                
        return "overall"

    def _extract_time_period(self, user_input: str) -> str:
        """Extract time period from user input."""
        input_lower = user_input.lower()
        
        if any(period in input_lower for period in ["today", "day"]):
            return "day"
        elif any(period in input_lower for period in ["week", "weekly"]):
            return "week"
        elif any(period in input_lower for period in ["month", "monthly"]):
            return "month"
        else:
            return "week"  # Default

    def _extract_query_type(self, user_input: str) -> str:
        """Extract query type for suggestions."""
        input_lower = user_input.lower()
        
        type_keywords = {
            "network": ["network", "connection", "traffic", "ip"],
            "process": ["process", "executable", "program", "application"],
            "file": ["file", "document", "path", "directory"],
            "user": ["user", "account", "login", "authentication"],
            "malware": ["malware", "virus", "threat", "suspicious"],
            "security": ["security", "incident", "alert", "breach"]
        }
        
        for query_type, keywords in type_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                return query_type
                
        return "general"

    def _extract_correction_info(self, user_input: str) -> Dict[str, str]:
        """Extract correction information from user input."""
        # This is a simplified version - you might want to enhance this
        # to better parse correction statements
        
        correction_patterns = [
            r"original:\s*(.+?)\s*->.*was:\s*(.+?)\s*->.*should.*be:\s*(.+?)(?:\s|$)",
            r"was:\s*(.+?)\s*should.*be:\s*(.+?)(?:\s|$)",
            r"correct.*query.*is:\s*(.+?)(?:\s|$)"
        ]
        
        for pattern in correction_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    return {
                        "original_nl": groups[0].strip(),
                        "original_query": groups[1].strip(),
                        "corrected_query": groups[2].strip(),
                        "feedback": "User correction"
                    }
                elif len(groups) >= 2:
                    return {
                        "original_nl": "Previous query",
                        "original_query": groups[0].strip(),
                        "corrected_query": groups[1].strip(),
                        "feedback": "User correction"
                    }
        
        return {}

    def _format_session_memories(self, memories: List[Dict]) -> str:
        """Format session-specific memories."""
        if not memories:
            return "No relevant memories found in this session."
        
        formatted = "Found in this session:\n\n"
        for i, memory in enumerate(memories[:3], 1):
            timestamp = memory.get('timestamp', 'Unknown time')
            content = memory.get('content', str(memory))[:200]
            formatted += f"{i}. **{timestamp}**: {content}...\n\n"
        
        return formatted

    def _format_conversion_response(self, user_input: str, converted_query: str) -> str:
        """Format conversion response."""
        return f"{converted_query}"

    def _format_session_history_response(self, state: "AssistantState") -> str:
        """Format session history response."""
        session_summary = state.get("session_summary", {})
        conversations = state.get("session_conversations", [])
        
        response = "Your Session History:\n\n"
        
        if session_summary and session_summary.get("conversation_count", 0) > 0:
            response += f"**Summary:**\n"
            response += f"• Total conversations: {session_summary['conversation_count']}\n"
            response += f"• Session started: {session_summary.get('first_conversation', 'Unknown')}\n"
            response += f"• Duration: {session_summary.get('duration_hours', 0)} hours\n\n"
            
            if conversations:
                response += "**Recent Conversations:**\n"
                for i, conv in enumerate(conversations[-5:], 1):
                    if isinstance(conv, HumanMessage):
                        content = conv.content[:50]
                        response += f"{i}. You: {content}...\n"
                    elif isinstance(conv, AIMessage):
                        content = conv.content[:50] 
                        response += f"{i}. Assistant: {content}...\n"
        else:
            response += "No previous conversations found in this session."
        
        return response

    def _generate_fallback_response(self, user_input: str, task: str) -> str:
        """Generate fallback response when task processing fails."""
        fallback_responses = {
            "conversion": "I can help convert your natural language query to a structured format. Please try rephrasing your request.",
            "memory_search": "I can search through previous conversations and queries. Please specify what you're looking for.",
            "analysis": "I can analyze query patterns and provide insights. Please specify what type of analysis you need.",
            "suggestions": "I can provide query suggestions and examples. Please specify what domain you're interested in.",
            "session_history": "I can show your session history. There may not be much history available yet.",
            "correction": "I can record corrections to improve future conversions. Please provide the original and corrected versions.",
        }
        
        return fallback_responses.get(task, "I'm here to help with security queries. Please let me know what you need assistance with.")

    def _generate_error_response(self, error_msg: str, task: str) -> str:
        """Generate user-friendly error responses."""
        # Don't expose technical error details to users
        friendly_responses = {
            "conversion": "I encountered an issue converting your query. Please try rephrasing it or providing more details.",
            "memory_search": "I couldn't search memories right now. Please try again or rephrase your search.",
            "analysis": "I couldn't complete the analysis at this time. Please try again later.",
            "suggestions": "I couldn't generate suggestions right now. Please try again or be more specific.",
            "session_history": "I couldn't retrieve your session history. Please try again.",
            "correction": "I couldn't record the correction. Please provide clear before/after examples.",
        }
        
        base_response = friendly_responses.get(task, "I encountered an issue processing your request.")
        return f"Error: {base_response} If the problem persists, please contact support."