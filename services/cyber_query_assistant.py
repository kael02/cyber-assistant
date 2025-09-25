from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
from functools import lru_cache
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI  
from memori import ConfigManager, Memori, create_memory_tool

from config import get_settings, logger
from services.document_ingestor import DocumentIngestor
from models import AssistantState
from tools import CyberQueryManager
from workflows import CyberQueryWorkflow
from prompts import SystemPrompts
from memory_system import MemoryManager


class CyberQueryAssistant:
    """LangGraph-based Cyber Query Assistant - WORKFLOW-ONLY ARCHITECTURE with OpenRouter."""

    def __init__(self):
        """Initialize with workflow-centric design using OpenRouter."""
        logger.info("🚀 Starting workflow-only initialization with OpenRouter...")

        try:
            # Core settings only
            self.settings = get_settings()
            self._db_url = self._build_db_url()
            
            # Initialize only essential components
            self._initialize_memory_system()
            self._initialize_llms()
            
            # Workflow will be initialized when first needed
            self._workflow_initialized = False
            self._graph = None
            self._memory_saver = None
            
            # Simple caching for responses only (not tool results)
            self._cache = {}
            self._max_cache_size = 50

            logger.info("✅ Workflow-only initialization with OpenRouter completed")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}", exc_info=True)
            raise

    def _build_db_url(self) -> str:
        """Build database URL once."""
        return (
            f"postgresql+psycopg2://{self.settings.DATABASE_USER}:"
            f"{self.settings.DATABASE_PASSWORD}@{self.settings.DATABASE_HOST}:"
            f"{self.settings.DATABASE_PORT}/{self.settings.DATABASE_NAME}"
        )

    def _initialize_memory_system(self):
        """Initialize memory system with optimized settings."""
        logger.info("🧠 Initializing optimized Memori...")
        
        # Use OpenRouter model for memory system if specified
        memory_model = getattr(self.settings, 'OPENROUTER_MEMORY_MODEL', 'openai/gpt-4o-mini')
        
        self.memory_system = Memori(
            database_connect=self._db_url,
            conscious_ingest=True,
            auto_ingest=False,  # Disabled for performance
            verbose=False,
            model=memory_model,
            # Add OpenRouter config for Memori if it supports it
            **self._get_openrouter_config_for_memori()
        )
        
        self.memory_system.enable()
        self.memory_tool = create_memory_tool(self.memory_system)
        logger.info("✅ Memory system ready with OpenRouter")

    def _get_openrouter_config_for_memori(self) -> Dict[str, Any]:
        """Get OpenRouter configuration for Memori if supported."""
        config = {}
        
        # Check if Memori supports OpenRouter configuration
        if hasattr(self.settings, 'OPENROUTER_API_KEY'):
            # Some versions of Memori might support custom API configurations
            config.update({
                'api_key': self.settings.OPENROUTER_API_KEY,
                'base_url': getattr(self.settings, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
            })
        
        return config

    def _initialize_llms(self):
        """Initialize LLMs with OpenRouter configuration."""
        logger.info("🤖 Initializing LLMs with OpenRouter...")
        
        # OpenRouter configuration
        openrouter_config = {
            "openai_api_key": self.settings.OPENROUTER_API_KEY,
            "openai_api_base": getattr(self.settings, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            "temperature": self.settings.LLM_TEMPERATURE,
            "timeout": 20,
            "max_retries": 1,
        }
        
        # Add custom headers for OpenRouter
        headers = {
            "HTTP-Referer": getattr(self.settings, 'OPENROUTER_REFERER', 'https://your-app.com'),
            "X-Title": getattr(self.settings, 'OPENROUTER_APP_NAME', 'CyberQuery Assistant'),
        }
        
        # Primary LLM with OpenRouter
        primary_model = getattr(self.settings, 'OPENROUTER_PRIMARY_MODEL', self.settings.LLM_MODEL)
        self.llm = ChatOpenAI(
            model=primary_model,
            default_headers=headers,
            **openrouter_config
        )
        
        # Finetuned/Secondary LLM with OpenRouter
        secondary_model = getattr(self.settings, 'OPENROUTER_SECONDARY_MODEL', self.settings.FINETUNED_MODEL_NAME)
        self.finetuned_llm = ChatOpenAI(
            model=secondary_model,
            default_headers=headers,
            **openrouter_config
        )
        
        logger.info(f"✅ LLMs ready with OpenRouter (Primary: {primary_model}, Secondary: {secondary_model})")

    def _initialize_workflow(self):
        """Initialize the complete workflow with all components."""
        if not self._workflow_initialized:
            logger.info("📊 Initializing complete workflow system...")
            
            # Initialize all components that workflow needs
            self.system_prompts = SystemPrompts()
            self.field_store = DocumentIngestor()
            self.field_context_k = getattr(self.settings, "FIELD_CONTEXT_K", 5)
            self.memory_manager = MemoryManager(self.memory_system)
            
            # Create tools manager (workflow will use this)
            self.tools_manager = CyberQueryManager(
                memory_tool=self.memory_tool,
                memory_system=self.memory_manager,
                field_store=self.field_store,
                field_context_k=self.field_context_k,
                llm=self.llm,
                finetuned_llm=self.finetuned_llm,
                system_prompts=self.system_prompts
            )
            
            # Initialize memory saver
            self.memory_saver = MemorySaver()
            
            # Create workflow (this will handle all tool interactions)
            self.workflow_manager = CyberQueryWorkflow(
                tools_manager=self.tools_manager,
                memory_system=self.memory_manager,
                system_prompts=self.system_prompts,
                llm=self.llm
            )
            
            # Create the executable graph
            self._graph = self.workflow_manager.create_graph(self.memory_saver)
            self._workflow_initialized = True
            logger.info("✅ Complete workflow system ready with OpenRouter")

    def _get_from_cache(self, key: str) -> Optional[str]:
        """Simple cache retrieval."""
        return self._cache.get(key)

    def _add_to_cache(self, key: str, value: str):
        """Simple cache storage with size limit."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    @lru_cache(maxsize=50)
    def _get_cache_key(self, user_input: str, session_id: str, task_type: str = "general") -> str:
        """Generate cache key with task type for better granularity."""
        return f"{hash(user_input)}_{session_id}_{task_type}"

    def _detect_task_type(self, user_input: str) -> str:
        """Detect the primary task type for better caching and routing."""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["convert", "translate", "query", "find logs", "search for"]):
            return "conversion"
        elif any(word in input_lower for word in ["remember", "recall", "past", "search memories"]):
            return "memory_search"
        elif any(word in input_lower for word in ["analyze", "pattern", "trends", "accuracy"]):
            return "analysis"
        elif any(word in input_lower for word in ["suggest", "example", "help", "how to"]):
            return "suggestions"
        elif any(word in input_lower for word in ["history", "previous", "earlier", "conversation"]):
            return "session_history"
        else:
            return "general"

    def process_user_input(self, user_input: str, session_id: str = "default", conversation_history: list = None) -> str:
        """WORKFLOW-ONLY processing with structured output integration."""
        logger.info(f"Processing via OpenRouter workflow: {user_input[:50]}... (session: {session_id})")

        try:
            start_time = datetime.now()

            # Detect task type for better caching
            task_type = self._detect_task_type(user_input)

            # Quick cache check
            cache_key = self._get_cache_key(user_input, session_id, task_type)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.info("Cache hit")
                return cached_response

            # Ensure workflow is ready
            self._initialize_workflow()

            # Build initial state with structured output context
            initial_state = AssistantState(
                messages=[HumanMessage(content=user_input)],
                user_input=user_input,
                mode="auto",
                current_task=task_type,
                memory_context="",
                conversion_result="",
                analysis_result="",
                error_message="",
                tool_calls=[],
                session_id=session_id,
                # Add structured output context
                conversation_history=conversation_history or [],
                metadata = {
                    "from_timestamp": None,
                    "to_timestamp": None,
                    "query_type": None,
                    "confidence": None
                }
            )

            # Process through workflow with structured output config
            config = RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "use_structured_output": True
                },
                recursion_limit=10
            )
            
            final_state = self._graph.invoke(initial_state, config)
            logger.info(f"Final state: {final_state}")
            # Extract response from workflow
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                duration = (datetime.now() - start_time).total_seconds()
                response = {
                    "response": ai_messages[-1].content,
                    "session_id": final_state["session_id"],
                    "timestamp": datetime.now().isoformat(),
                    "duration": duration,
                    "from_timestamp": final_state["metadata"]["from_timestamp"],
                    "to_timestamp": final_state["metadata"]["to_timestamp"],
                    "confidence": final_state["metadata"]["confidence"],
                    "task_type": final_state["current_task"],
                    "success": "success" if final_state["current_task"] != "error" else "error",
                    "model_provider": "openrouter",
                }
                
                # Cache the response
                self._add_to_cache(cache_key, response)
                
                logger.info(f"OpenRouter workflow completed in {duration:.2f}s")
                return response
            else:
                return "I processed your request through the OpenRouter workflow but couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"OpenRouter workflow processing error: {str(e)}", exc_info=True)
            return "I encountered an error. Please try rephrasing your request or contact support."

    # SIMPLIFIED API METHODS - All route through workflow
    def chat_with_memory(self, user_input: str, session_id: str = "api") -> str:
        """Chat with memory - routes through workflow."""
        return self.process_user_input(user_input, session_id=session_id)

    def convert_to_query(self, natural_language: str, context: str = "", session_id: str = "convert") -> str:
        """Convert to query - routes through workflow."""
        # Format as a conversion request
        request = f"{natural_language}"
        if context:
            request += f" (Context: {context})"
        
        return self.process_user_input(request, session_id=session_id)

    def analyze_query_patterns(self, analysis_type: str = "overall", time_period: str = "month", session_id: str = "analyze") -> str:
        """Analyze patterns - routes through workflow."""
        request = f"Analyze {analysis_type} query patterns for the past {time_period}"
        return self.process_user_input(request, session_id=session_id)

    def get_query_suggestions(self, query_type: str = "general", session_id: str = "suggest") -> str:
        """Get suggestions - routes through workflow."""
        request = f"Give me query suggestions for {query_type} security analysis"
        return self.process_user_input(request, session_id=session_id)

    def memory_search(self, query: str, session_id: str = "memory") -> str:
        """Search memory - routes through workflow."""
        request = f"Search my memories for: {query}"
        return self.process_user_input(request, session_id=session_id)

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, feedback: str = "", session_id: str = "correction") -> str:
        """Record correction - routes through workflow."""
        request = f"Record this correction - Original: {original_nl} -> Was: {original_query} -> Should be: {corrected_query}"
        if feedback:
            request += f" (Feedback: {feedback})"
        
        return self.process_user_input(request, session_id=session_id)

    def get_conversion_summary(self, session_id: str = "summary") -> str:
        """Get conversion summary - routes through workflow."""
        request = "Give me a summary of recent query conversions"
        return self.process_user_input(request, session_id=session_id)

    def get_session_history(self, session_id: str, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get session history - routes through workflow."""
        try:
            # This is a special case - we need to access the memory system
            # But we'll do it through a workflow call
            request = f"Show me the history for this session (last {max_messages} messages)"
            response = self.process_user_input(request, session_id=session_id)
            
            # The workflow should return structured data, but for now return formatted response
            # You might want to enhance the workflow to return structured data for this case
            return [
                {
                    'type': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
        except Exception as e:
            logger.error(f"❌ Session history error: {str(e)}")
            return []

    def clear_session_history(self, session_id: str) -> bool:
        """Clear session history."""
        try:
            if hasattr(self, '_memory_saver') and self._memory_saver:
                config = RunnableConfig(configurable={"thread_id": session_id})
                self._memory_saver.delete(config)
                logger.info(f"🧹 Cleared session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Clear session error: {str(e)}")
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get basic performance stats."""
        return {
            'cache_size': len(self._cache),
            'workflow_initialized': self._workflow_initialized,
            'memory_system_active': hasattr(self, 'memory_system') and self.memory_system is not None,
            'model_provider': 'openrouter',
            'primary_model': getattr(self.settings, 'OPENROUTER_PRIMARY_MODEL', 'unknown'),
            'secondary_model': getattr(self.settings, 'OPENROUTER_SECONDARY_MODEL', 'unknown')
        }

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("🧹 Cache cleared")

    def trigger_memory_analysis(self):
        """Trigger memory analysis."""
        try:
            if hasattr(self, 'memory_system') and self.memory_system:
                self.memory_system.trigger_conscious_analysis()
                logger.info("✅ Memory analysis completed")
        except Exception as e:
            logger.error(f"❌ Memory analysis failed: {str(e)}")

    def warmup(self, sample_queries: List[str] = None):
        """Warm up the system."""
        logger.info("🔥 Warming up OpenRouter workflow...")
        
        # Initialize workflow
        self._initialize_workflow()
        
        # Run sample queries through workflow
        default_queries = [
            "Convert this: show me failed login attempts",
            "What query suggestions do you have?"
        ]
        
        queries_to_run = sample_queries or default_queries
        
        for query in queries_to_run[:2]:
            try:
                self.process_user_input(query, session_id="warmup")
                logger.info(f"✅ Warmup query completed: {query[:30]}...")
            except Exception as e:
                logger.warning(f"⚠️ Warmup failed for query: {str(e)}")
        
        logger.info("✅ OpenRouter workflow warmup completed")

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models (if supported by your settings)."""
        try:
            # This would depend on your OpenRouter integration
            # You might want to implement a method to query available models
            return getattr(self.settings, 'OPENROUTER_AVAILABLE_MODELS', [
                'openai/gpt-4o',
                'openai/gpt-4o-mini',
                'anthropic/claude-3-sonnet',
                'anthropic/claude-3-haiku',
                'google/gemini-pro',
                'meta-llama/llama-3-70b-instruct'
            ])
        except Exception as e:
            logger.error(f"❌ Error getting available models: {str(e)}")
            return []

    def switch_model(self, model_name: str, model_type: str = "primary") -> bool:
        """Switch the primary or secondary model dynamically."""
        try:
            openrouter_config = {
                "openai_api_key": self.settings.OPENROUTER_API_KEY,
                "openai_api_base": getattr(self.settings, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                "temperature": self.settings.LLM_TEMPERATURE,
                "timeout": 20,
                "max_retries": 1,
            }
            
            headers = {
                "HTTP-Referer": getattr(self.settings, 'OPENROUTER_REFERER', 'https://your-app.com'),
                "X-Title": getattr(self.settings, 'OPENROUTER_APP_NAME', 'CyberQuery Assistant'),
            }
            
            new_llm = ChatOpenAI(
                model=model_name,
                default_headers=headers,
                **openrouter_config
            )
            
            if model_type == "primary":
                self.llm = new_llm
                logger.info(f"✅ Switched primary model to: {model_name}")
            else:
                self.finetuned_llm = new_llm
                logger.info(f"✅ Switched secondary model to: {model_name}")
            
            # Reset workflow to use new models
            self._workflow_initialized = False
            return True
            
        except Exception as e:
            logger.error(f"❌ Model switch failed: {str(e)}")
            return False