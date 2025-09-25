import re
from typing import List, Dict, Any 
from langchain_core.messages import HumanMessage , AIMessage
from config import logger


class ConversationContextManager:
    """Manages conversation context and continuation detection."""
    
    def build_conversation_context(self, conversation_history: list, max_messages: int = 4) -> str:
        """Build contextual information from recent conversation history."""
        if not conversation_history:
            return ""
        
        try:
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

    def extract_previous_query_context(self, conversation_history: list) -> Dict[str, Any]:
        """Extract context from previous queries to understand follow-up intent."""
        if not conversation_history:
            return self._empty_context()
        
        context = self._empty_context()
        
        try:
            for msg in reversed(conversation_history):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    
                    if self._contains_query(content):
                        context['has_previous_query'] = True
                        context['previous_query'] = content
                        context['previous_conditions'] = self._extract_query_conditions(content)
                        break
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return context

    def detect_continuation_intent(self, user_input: str, conversation_history: list) -> Dict[str, Any]:
        """Detect if the user input is a continuation of a previous query."""
        continuation_info = {
            'is_continuation': False,
            'conjunction_type': 'AND',
            'new_conditions': [],
            'modification_type': 'add'
        }
        
        user_lower = user_input.lower().strip()
        
        # Check explicit continuation patterns
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
                continuation_info.update({
                    'is_continuation': True,
                    'conjunction_type': conjunction,
                    'modification_type': modification,
                    'new_conditions': [match.group(1)]
                })
                break
        
        # Check for implicit continuation
        if not continuation_info['is_continuation'] and conversation_history:
            previous_context = self.extract_previous_query_context(conversation_history)
            if (previous_context['has_previous_query'] and 
                self._has_similar_field_references(user_input, previous_context['previous_conditions'])):
                continuation_info.update({
                    'is_continuation': True,
                    'new_conditions': [user_input]
                })
        
        return continuation_info

    def _empty_context(self) -> Dict[str, Any]:
        """Return empty context structure."""
        return {
            'has_previous_query': False,
            'previous_query': '',
            'previous_conditions': [],
            'query_type': '',
            'continuation_indicators': []
        }

    def _contains_query(self, content: str) -> bool:
        """Check if content contains a generated query."""
        return ('```' in content or 
                any(keyword in content.lower() for keyword in ['source_ip:', 'destination_ip:', 'SELECT', 'WHERE']))

    def _extract_query_conditions(self, query_text: str) -> List[str]:
        """Extract conditions from a previously generated query."""
        conditions = []
        
        try:
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