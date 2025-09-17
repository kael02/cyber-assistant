from ast import mod
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import concurrent.futures
from functools import lru_cache
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from memori import ConfigManager, Memori, create_memory_tool

from config import get_settings, logger
from services.document_ingestor import DocumentIngestor
from models import AssistantState
from tools import CyberQueryTools
from workflows import CyberQueryWorkflow
from prompts import SystemPrompts


class CyberQueryAssistant:
    """LangGraph-based Cyber Query Assistant with advanced memory and workflow orchestration - PERFORMANCE OPTIMIZED."""

    def __init__(self):
        """Initialize the LangGraph Security Query Assistant."""
        logger.info("🚀 Starting CyberQueryAssistant initialization...")

        try:
            self._initialize_settings()
            self._initialize_database_config()
            self._initialize_field_store()
            self._initialize_memory_system()
            self._initialize_llms()
            self._initialize_tools()
            self._initialize_workflow()
            
            # Performance monitoring
            self._query_cache = {}
            self._cache_max_size = 100
            self._performance_metrics = {
                'total_queries': 0,
                'avg_response_time': 0.0,
                'cache_hits': 0
            }

            logger.info("🎉 CyberQueryAssistant initialization completed successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize CyberQueryAssistant: {str(e)}", exc_info=True)
            raise

    def _initialize_settings(self):
        """Initialize application settings."""
        self.settings = get_settings()
        logger.info("✅ Settings loaded successfully")

    def _initialize_database_config(self):
        """Initialize database configuration."""
        self.database_user = self.settings.DATABASE_USER
        self.database_password = self.settings.DATABASE_PASSWORD
        self.database_host = self.settings.DATABASE_HOST
        self.database_port = self.settings.DATABASE_PORT
        self.database_name = self.settings.DATABASE_NAME

        logger.info(f"🔧 Database config: {self.database_user}@{self.database_host}:{self.database_port}/{self.database_name}")

    def _initialize_field_store(self):
        """Initialize field semantic search store."""
        logger.info("📚 Initializing DocumentIngestor for field semantic context...")
        self.field_store = DocumentIngestor()
        self.field_context_k = getattr(self.settings, "FIELD_CONTEXT_K", 5)  # Reduced from 8 to 5 for performance
        logger.info("✅ Field catalog semantic search ready")

    def _initialize_memory_system(self):
        """Initialize Memori memory system with CORRECTED performance optimizations."""
        logger.info("🧠 Initializing Memori with PostgreSQL database...")
        db_url = f"postgresql+psycopg2://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
        
        # CORRECTED PERFORMANCE OPTIMIZATION based on actual Memori docs
        self.memory_system = Memori(
            database_connect=db_url,
            conscious_ingest=True,     # ✅ KEEP TRUE - One-shot working memory (efficient)
            auto_ingest=True,         # ✅ DISABLE - This causes continuous DB searches (slow)
            verbose=False,             # ✅ Reduce logging overhead
            model='gpt-4o-mini',
        )
        
        # CRITICAL FIX: Actually enable the memory system!
        logger.info("🔄 Enabling Memori memory system...")
        self.memory_system.enable()
        logger.info("✅ Memori enabled successfully")
        
        logger.info("📝 Creating memory tool...")
        self.memory_tool = create_memory_tool(self.memory_system)
        logger.info("✅ Memory tool created")

    def _initialize_llms(self):
        """Initialize language models."""
        logger.info("🤖 Initializing LLMs...")
        
        # PERFORMANCE OPTIMIZATION: Reduced timeouts
        self.llm = ChatOpenAI(
            model=self.settings.LLM_MODEL,
            temperature=self.settings.LLM_TEMPERATURE,
            timeout=30,  # Reduced from default 60s
            max_retries=1,  # Reduced retries for faster failures
        )
        logger.info(f"✅ Main LLM initialized: {self.settings.LLM_MODEL}")

        self.finetuned_llm = ChatOpenAI(
            model=self.settings.FINETUNED_MODEL_NAME,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=self.settings.LLM_TEMPERATURE,
            timeout=30,  # Reduced from 60
            max_retries=1,  # Reduced retries
        )
        logger.info(f"✅ Fine-tuned LLM initialized: {self.settings.FINETUNED_MODEL_NAME}")

    def _initialize_tools(self):
        """Initialize tools."""
        logger.info("🛠️ Creating tools...")
        self.system_prompts = SystemPrompts()
        
        self.tools_manager = CyberQueryTools(
            memory_tool=self.memory_tool,
            memory_system=self.memory_system,
            field_store=self.field_store,
            field_context_k=self.field_context_k,
            llm=self.llm,
            finetuned_llm=self.finetuned_llm,
            system_prompts=self.system_prompts
        )
        
        self.tools = self.tools_manager.create_tools()
        logger.info(f"✅ {len(self.tools)} tools created successfully")

    def _initialize_workflow(self):
        """Initialize LangGraph workflow."""
        logger.info("📊 Creating LangGraph workflow...")
        
        self.memory_saver = MemorySaver()
        
        self.workflow_manager = CyberQueryWorkflow(
            tools_manager=self.tools_manager,
            memory_system=self.memory_system,
            system_prompts=self.system_prompts,
            llm=self.llm
        )
        
        self.graph = self.workflow_manager.create_graph(self.memory_saver)
        logger.info("✅ LangGraph workflow created and compiled")

    @lru_cache(maxsize=50)
    def _get_cache_key(self, user_input: str, session_id: str) -> str:
        """Generate cache key for query responses."""
        return f"{hash(user_input)}_{session_id}"

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        if cache_key in self._query_cache:
            self._performance_metrics['cache_hits'] += 1
            logger.info(f"🎯 Cache hit for query")
            return self._query_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache response with size limit."""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = response

    def _update_performance_metrics(self, duration: float):
        """Update performance tracking metrics."""
        self._performance_metrics['total_queries'] += 1
        total_queries = self._performance_metrics['total_queries']
        current_avg = self._performance_metrics['avg_response_time']
        
        # Update rolling average
        self._performance_metrics['avg_response_time'] = (
            (current_avg * (total_queries - 1) + duration) / total_queries
        )

    def process_user_input(self, user_input: str, session_id: str = "default") -> str:
        """Process user input through the LangGraph workflow."""
        logger.info(f"🎯 Processing user input for session '{session_id}': '{user_input[:100]}...'")

        try:
            start_time = datetime.now()

            # Check cache first for performance
            cache_key = self._get_cache_key(user_input, session_id)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.info("⚡ Returning cached response")
                return cached_response

            initial_state = AssistantState(
                messages=[HumanMessage(content=user_input)],
                user_input=user_input,
                mode="converter",
                current_task="",
                memory_context="",
                conversion_result="",
                analysis_result="",
                error_message="",
                tool_calls=[],
                session_id=session_id
            )

            config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = self.graph.invoke(initial_state, config)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
                
                # Cache the response for future use
                self._cache_response(cache_key, response)
                
                # Update performance metrics
                self._update_performance_metrics(duration)
                
                logger.info(f"✅ Processing completed in {duration:.2f}s, response length: {len(response)}")
                return response
            else:
                logger.warning("⚠️ No AI messages found in final state")
                return "I processed your request, but couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"❌ Error processing user input: {str(e)}", exc_info=True)
            return f"Error processing your request: {str(e)}"

    # Public API methods
    def chat_with_memory(self, user_input: str, session_id: str = "api") -> str:
        """Chat with memory capabilities."""
        return self.process_user_input(user_input, session_id=session_id)

    def convert_to_query(self, natural_language: str, context: str = "") -> str:
        """Convert natural language to structured query."""
        # Add caching for conversions
        cache_key = f"convert_{hash(natural_language)}_{hash(context)}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
            
        result = self.tools_manager.convert_to_query_tool(natural_language, context)
        self._cache_response(cache_key, result)
        return result

    def analyze_query_patterns(self, analysis_type: str = "overall", time_period: str = "month") -> str:
        """Analyze query patterns."""
        return self.tools_manager.analyze_patterns_tool(analysis_type, time_period)

    def get_query_suggestions(self, query_type: str = "general") -> str:
        """Get query suggestions."""
        return self.tools_manager.get_suggestions_tool(query_type)

    def memory_search(self, query: str) -> str:
        """Search memory for relevant information."""
        return self.tools_manager.memory_search_tool(query)

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, feedback: str = "") -> str:
        """Record user corrections to improve future conversions."""
        return self.tools_manager.record_correction(original_nl, original_query, corrected_query, feedback)

    def get_conversion_summary(self) -> str:
        """Generate a summary of today's query conversions."""
        return self.tools_manager.get_conversion_summary()

    # Additional performance methods
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            **self._performance_metrics,
            'cache_size': len(self._query_cache),
            'cache_hit_rate': (
                self._performance_metrics['cache_hits'] / max(1, self._performance_metrics['total_queries'])
            ) * 100
        }

    def clear_cache(self):
        """Clear response cache to free memory."""
        self._query_cache.clear()
        logger.info("🧹 Response cache cleared")

    def trigger_memory_analysis(self):
        """Manually trigger Memori conscious analysis for better context."""
        logger.info("🧠 Triggering manual memory analysis...")
        try:
            self.memory_system.trigger_conscious_analysis()
            logger.info("✅ Memory analysis completed")
        except Exception as e:
            logger.error(f"❌ Memory analysis failed: {str(e)}")

    def warmup(self, sample_queries: List[str] = None):
        """Warm up the system by running sample queries."""
        logger.info("🔥 Warming up system...")
        
        # Trigger memory analysis first
        self.trigger_memory_analysis()
        
        # Run sample queries if provided
        if sample_queries:
            for query in sample_queries[:2]:  # Limit warmup queries
                try:
                    self.process_user_input(query, session_id="warmup")
                    logger.info(f"✅ Warmup query completed: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️ Warmup query failed: {str(e)}")
        
        logger.info("✅ System warmup completed")