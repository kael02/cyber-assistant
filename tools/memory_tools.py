from datetime import datetime
from typing import List, Dict, Optional
from config import logger
from langchain_core.messages import SystemMessage, HumanMessage
from .contexts.conversation_context import ConversationContextManager


class MemoryTools:
    """Handles memory search and pattern analysis operations."""
    
    def __init__(self, memory_tool, memory_system, llm):
        self.memory_tool = memory_tool
        self.memory_system = memory_system
        self.llm = llm

    def search_memory(self, query: str, session_id: str = None) -> str:
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

    def analyze_patterns(self, analysis_type: str, time_period: str = "week", 
                        conversation_history: list = None, session_id: str = None) -> str:
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
                memories = self.search_memory(query, session_id=session_id)
            else:
                memories = self.search_memory(query)

            # Build conversation context
            context_manager = ConversationContextManager()
            conv_context = context_manager.build_conversation_context(conversation_history or [])

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