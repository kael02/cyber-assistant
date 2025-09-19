from typing import List, Any, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from config import logger
import json

class MemoryManager:
    """Simple and reliable memory manager using LangGraph's MemorySaver."""
    
    def __init__(self, memori_instance=None):
        self.memori = memori_instance  # Keep for compatibility but don't rely on it
        self.memory_saver = MemorySaver()  # Use LangGraph's built-in memory
        
        # Simple in-memory backup for when MemorySaver isn't available
        self._session_store = {}
        self._cleanup_interval = 100  # Clean up every 100 operations
        self._operation_count = 0
    
    def get_session_context(self, session_id: str, max_messages: int = 10) -> List[Any]:
        """Get session context using LangGraph's MemorySaver."""
        try:
            logger.info(f"Getting session context for session {session_id}")
            from langchain_core.runnables import RunnableConfig
            
            # Create config for this session
            config = RunnableConfig(
                configurable={"thread_id": session_id}
            )
            
            # Try to get checkpoint from MemorySaver
            checkpoint = self.memory_saver.get(config)
            
            if checkpoint and 'channel_values' in checkpoint:
                # Extract messages from checkpoint
                channel_values = checkpoint['channel_values']
                
                # Look for messages in the channel values
                if 'messages' in channel_values:
                    messages = channel_values['messages']
                    
                    # Convert to proper LangChain message format if needed
                    formatted_messages = []
                    for msg in messages[-max_messages:]:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            formatted_messages.append(msg)
                        elif hasattr(msg, 'content'):
                            # Try to determine message type
                            if hasattr(msg, 'type'):
                                if msg.type == 'human':
                                    formatted_messages.append(HumanMessage(content=msg.content))
                                elif msg.type == 'ai':
                                    formatted_messages.append(AIMessage(content=msg.content))
                    
                    logger.debug(f"Retrieved {len(formatted_messages)} messages from MemorySaver for session {session_id}")
                    return formatted_messages
            
            # Fallback to simple in-memory store
            return self._get_fallback_context(session_id, max_messages)
            
        except Exception as e:
            logger.debug(f"MemorySaver failed, using fallback: {e}")
            return self._get_fallback_context(session_id, max_messages)
    
    def record_session_interaction(self, session_id: str, user_input: str, assistant_response: str):
        """Record session interaction in both MemorySaver and fallback store."""
        try:
            # Record in simple fallback store (always works)
            self._record_fallback_interaction(session_id, user_input, assistant_response)
            
            # Note: LangGraph's workflow will handle MemorySaver automatically
            # We don't need to manually save to MemorySaver here
            
            logger.debug(f"Session {session_id} interaction recorded")
            
        except Exception as e:
            logger.error(f"Session recording failed: {e}")
    
    def search_session_memories(self, session_id: str, search_query: str, limit: int = 5) -> List[Dict]:
        """Search within session using simple text matching."""
        try:
            # Get session context
            messages = self.get_session_context(session_id, max_messages=50)
            
            search_terms = search_query.lower().split()
            results = []
            
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content.lower()
                    
                    # Simple relevance scoring
                    score = sum(1 for term in search_terms if term in content)
                    
                    if score > 0:
                        msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        results.append({
                            'content': f"{msg_type}: {msg.content[:200]}",
                            'timestamp': datetime.now().isoformat(),
                            'type': msg_type.lower(),
                            'relevance_score': score
                        })
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.debug(f"Found {len(results[:limit])} session memories for query: {search_query}")
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Session memory search failed: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary from available data."""
        try:
            messages = self.get_session_context(session_id, max_messages=100)
            
            if not messages:
                return self._empty_summary()
            
            # Count user messages (conversations)
            user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            
            # Estimate timing (since we don't have exact timestamps)
            now = datetime.now()
            estimated_start = now - timedelta(hours=1)  # Assume 1 hour session max
            
            return {
                'conversation_count': len(user_messages),
                'total_messages': len(messages),
                'first_conversation': estimated_start.isoformat(),
                'last_conversation': now.isoformat(),
                'duration_hours': 1.0  # Estimated
            }
            
        except Exception as e:
            logger.error(f"Session summary failed: {e}")
            return self._empty_summary()
    
    def clear_session_history(self, session_id: str) -> bool:
        """Clear session history from both stores."""
        try:
            success = True
            
            # Clear from MemorySaver
            try:
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(configurable={"thread_id": session_id})
                self.memory_saver.delete(config)
            except Exception as e:
                logger.debug(f"MemorySaver delete failed: {e}")
                success = False
            
            # Clear from fallback store
            if session_id in self._session_store:
                del self._session_store[session_id]
            
            logger.info(f"Session {session_id} history cleared")
            return success
            
        except Exception as e:
            logger.error(f"Failed to clear session history: {e}")
            return False
    
    def _get_fallback_context(self, session_id: str, max_messages: int) -> List[Any]:
        """Get context from simple in-memory fallback store."""
        session_data = self._session_store.get(session_id, {})
        messages_data = session_data.get('messages', [])
        
        # Convert to LangChain messages
        messages = []
        for msg_data in messages_data[-max_messages:]:
            if msg_data['type'] == 'human':
                messages.append(HumanMessage(content=msg_data['content']))
            elif msg_data['type'] == 'ai':
                messages.append(AIMessage(content=msg_data['content']))
        
        return messages
    
    def _record_fallback_interaction(self, session_id: str, user_input: str, assistant_response: str):
        """Record in simple fallback store."""
        if session_id not in self._session_store:
            self._session_store[session_id] = {
                'messages': [],
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        
        session_data = self._session_store[session_id]
        
        # Add messages
        session_data['messages'].extend([
            {
                'type': 'human',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'ai',
                'content': assistant_response,
                'timestamp': datetime.now().isoformat()
            }
        ])
        
        # Keep only recent messages
        session_data['messages'] = session_data['messages'][-50:]  # Keep last 50 messages
        session_data['last_updated'] = datetime.now()
        
        # Periodic cleanup
        self._operation_count += 1
        if self._operation_count >= self._cleanup_interval:
            self._cleanup_old_sessions()
            self._operation_count = 0
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions from fallback store."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep sessions for 24 hours
        sessions_to_remove = []
        
        for session_id, session_data in self._session_store.items():
            last_updated = session_data.get('last_updated', datetime.now())
            if last_updated < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove[:10]:  # Remove max 10 at a time
            del self._session_store[session_id]
        
        if sessions_to_remove:
            logger.debug(f"Cleaned up {len(sessions_to_remove[:10])} old sessions")
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure."""
        return {
            'conversation_count': 0,
            'total_messages': 0,
            'first_conversation': None,
            'last_conversation': None,
            'duration_hours': 0
        }
    
    # Keep Memori compatibility methods but make them optional
    def enable_memori_auto_recording(self):
        """Enable Memori auto-recording if available."""
        if self.memori:
            try:
                self.memori.enable()
                logger.info("Memori auto-recording enabled")
            except Exception as e:
                logger.debug(f"Memori enable failed: {e}")
    
    def trigger_conscious_analysis(self):
        """Trigger Memori conscious analysis if available."""
        if self.memori and hasattr(self.memori, 'trigger_conscious_analysis'):
            try:
                self.memori.trigger_conscious_analysis()
                logger.info("Memori conscious analysis triggered")
            except Exception as e:
                logger.debug(f"Memori conscious analysis failed: {e}")
    
    def get_memori_status(self) -> Dict[str, Any]:
        """Get status of memory systems."""
        return {
            'memori_available': self.memori is not None,
            'memory_saver_available': self.memory_saver is not None,
            'fallback_sessions': len(self._session_store),
            'total_operations': self._operation_count
        }