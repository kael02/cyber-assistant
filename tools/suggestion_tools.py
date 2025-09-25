from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from config import logger
from .contexts.conversation_context import ConversationContextManager


class SuggestionTools:
    """Handles query suggestions and examples."""
    
    def __init__(self, llm):
        self.llm = llm

    def get_suggestions(self, query_type: str, conversation_history: list = None, 
                       session_id: str = None) -> str:
        """Enhanced suggestions with conversation context."""
        logger.info(f"Suggestions for: {query_type}")
        
        # Build conversation context for dynamic suggestions
        context_manager = ConversationContextManager()
        conv_context = context_manager.build_conversation_context(conversation_history or [])
        
        # Pre-built suggestions for speed
        quick_suggestions = self._get_quick_suggestions()
        
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

    def _get_quick_suggestions(self) -> Dict[str, str]:
        """Get pre-built quick suggestions for common query types."""
        return {
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
