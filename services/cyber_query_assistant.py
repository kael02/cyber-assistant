
from datetime import date, datetime
from typing import List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from memori import Memori, create_memory_tool
from models import AssistantState
from config import get_settings, logger
class CyberQueryAssistant:
    """LangGraph-based Personal Diary Assistant with advanced memory and workflow orchestration."""

    def __init__(self):
        """Initialize the LangGraph Security Query Assistant."""
        self.settings = get_settings()
        self.database_user = self.settings.DATABASE_USER
        self.database_password = self.settings.DATABASE_PASSWORD
        self.database_host = self.settings.DATABASE_HOST
        self.database_port = self.settings.DATABASE_PORT
        self.database_name = self.settings.DATABASE_NAME
        logger.info("Initializing Memori with PostgreSQL database...")
        self.memory_system = Memori(
            database_connect=f"postgresql+psycopg2://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}",
            conscious_ingest=True,
            auto_ingest=True,
        )

        logger.info("Enabling memory tracking...")
        self.memory_system.enable()
        self.llm = ChatOpenAI(model=self.settings.LLM_MODEL, temperature=self.settings.LLM_TEMPERATURE)

        self.finetuned_llm = ChatOpenAI(
            model=self.settings.FINETUNED_MODEL_NAME,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=self.settings.LLM_TEMPERATURE,
            timeout=60,
        )

        # Create memory tool
        self.memory_tool = create_memory_tool(self.memory_system)

        # System prompts for different functionalities
        self.system_prompts = {
            "converter": """You are an expert security analyst and query translator with advanced memory capabilities.
            Your role is to help users convert natural language security queries into structured application queries
            like graylog format. On your output, you should only return the query, not any other text. Use memory_search to understand the user's query patterns,
            preferences, and past conversions. Be precise, learn from corrections, and provide accurate translations.""",
            
            "analyzer": """You are a security query optimization expert. Analyze the user's query patterns,
            conversion accuracy, and usage trends to provide insights. Use memory_search to gather comprehensive
            information about the user's query history. Provide specific recommendations for improving query
            accuracy and efficiency.""",
            
            "pattern_recognition": """You are a security pattern recognition specialist. Help users identify
            common query patterns, optimize their natural language inputs, and learn from their query history.
            Use memory_search to understand their past queries and suggest improvements.""",
        }

        # Initialize tools
        self.tools = self._create_tools()
        self.memory_saver = MemorySaver()
 
        # Create the graph
        self.graph = self._create_graph()
        

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the assistant."""
        
        def memory_search_tool(query: str) -> str:
            """Search memories for relevant information."""
            try:
                result = self.memory_tool.execute(query=query)
                return str(result) if result else "No relevant memories found."
            except Exception as e:
                return f"Memory search error: {str(e)}"

        def convert_to_query_tool(natural_language: str, context: str = "") -> str:
            """Convert natural language to structured query."""
            try:
                # Search for similar past queries for context
                similar_queries = memory_search_tool(f"similar queries: {natural_language[:50]}")
                
                # Call fine-tuned model
                messages = [
                    SystemMessage(content=self.system_prompts["converter"]),
                    HumanMessage(content=natural_language),
                ]
                
                converted_query = self.finetuned_llm.invoke(messages)
                
                # Store the conversion in memory
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                conversion_record = f"""
                Time: {current_time}
                Input: {natural_language}
                Generated Query: {converted_query.content}
                Context: {context}
                Similar Past Queries: {similar_queries}
                Status: {'Success' if not converted_query.content.startswith('Error:') else 'Failed'}
                """
                
                self.memory_system.record_conversation(
                    user_input=f"Convert query: {natural_language}",
                    ai_output=conversion_record
                )
                
                return converted_query.content
                
            except Exception as e:
                return f"Query conversion error: {str(e)}"

        def analyze_patterns_tool(analysis_type: str, time_period: str = "month") -> str:
            """Analyze patterns in user's query conversion history."""
            try:
                search_queries = {
                    "accuracy": f"query conversions corrections accuracy {time_period}",
                    "formats": f"output formats preferences {time_period}",
                    "complexity": f"complex queries simple queries {time_period}",
                    "topics": f"security topics network process file {time_period}",
                    "improvements": f"query improvements suggestions {time_period}",
                    "overall": f"conversion patterns trends {time_period}",
                }

                query = search_queries.get(analysis_type, f"{analysis_type} {time_period}")
                memories = memory_search_tool(query)

                analysis_prompt = f"""
                Based on the following query conversion history for {analysis_type} over the {time_period}:
                {memories}

                Please provide a comprehensive analysis including:
                1. Key patterns and trends in query conversions
                2. Most common output formats used
                3. Accuracy improvements over time
                4. Common mistakes or correction patterns
                5. Recommendations for better query formulation
                6. Areas for improvement
                """

                messages = [
                    SystemMessage(content=self.system_prompts["analyzer"]),
                    HumanMessage(content=analysis_prompt),
                ]

                response = self.llm.invoke(messages)
                return response.content

            except Exception as e:
                return f"Pattern analysis error: {str(e)}"

        def get_suggestions_tool(query_type: str) -> str:
            """Generate personalized query suggestions."""
            try:
                past_queries = memory_search_tool(f"{query_type} queries examples successful conversions")
                user_patterns = memory_search_tool(f"user preferences {query_type} query patterns")

                suggestion_prompt = f"""
                Based on the user's history with {query_type} queries:
                Past Queries: {past_queries}
                User Patterns: {user_patterns}

                Please provide personalized suggestions including:
                1. 5 example natural language queries for {query_type} security scenarios
                2. Best practices for formulating {query_type} queries
                3. Common fields and operators to use
                4. Tips to improve conversion accuracy
                5. Advanced query patterns for complex scenarios
                """

                messages = [
                    SystemMessage(content=self.system_prompts["pattern_recognition"]),
                    HumanMessage(content=suggestion_prompt),
                ]

                response = self.llm.invoke(messages)
                return response.content

            except Exception as e:
                return f"Suggestion error: {str(e)}"

        return [
            Tool(
                name="memory_search",
                description="Search memories for relevant information about past queries, patterns, corrections, and user preferences",
                func=memory_search_tool
            ),
            Tool(
                name="convert_to_query",
                description="Convert natural language to structured security query",
                func=convert_to_query_tool
            ),
            Tool(
                name="analyze_query_patterns",
                description="Analyze patterns in user's query conversion history",
                func=analyze_patterns_tool
            ),
            Tool(
                name="get_query_suggestions", 
                description="Generate personalized query suggestions based on user's history",
                func=get_suggestions_tool
            )
        ]

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        # Initialize the graph
        workflow = StateGraph(AssistantState)

        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("memory_search_node", self.memory_search_node)
        workflow.add_node("query_conversion_node", self.query_conversion_node)
        workflow.add_node("pattern_analysis_node", self.pattern_analysis_node)
        workflow.add_node("suggestion_node", self.suggestion_node)
        workflow.add_node("response_generation_node", self.response_generation_node)
        workflow.add_node("error_handling_node", self.error_handling_node)

        # Define the workflow edges
        workflow.add_edge(START, "classify_intent")
        
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_based_on_intent,
            {
                "memory_search": "memory_search_node",
                "query_conversion": "query_conversion_node", 
                "pattern_analysis": "pattern_analysis_node",
                "suggestions": "suggestion_node",
                "general": "response_generation_node",
                "error": "error_handling_node"
            }
        )
        
        workflow.add_edge("memory_search_node", "response_generation_node")
        workflow.add_edge("query_conversion_node", "response_generation_node")
        workflow.add_edge("pattern_analysis_node", "response_generation_node")
        workflow.add_edge("suggestion_node", "response_generation_node")
        workflow.add_edge("response_generation_node", END)
        workflow.add_edge("error_handling_node", END)

        return workflow.compile(checkpointer=self.memory_saver)

    def classify_intent_node(self, state: AssistantState) -> AssistantState:
        """Classify user intent to route to appropriate workflow."""
        user_input = state["user_input"].lower()
        
        # Intent classification logic
        if any(keyword in user_input for keyword in ["convert", "translate", "query", "find", "search for", "show me"]):
            if any(keyword in user_input for keyword in ["splunk", "elastic", "graylog", "format"]):
                state["current_task"] = "query_conversion"
            else:
                state["current_task"] = "query_conversion"
        elif any(keyword in user_input for keyword in ["analyze", "pattern", "trends", "accuracy", "performance"]):
            state["current_task"] = "pattern_analysis"
        elif any(keyword in user_input for keyword in ["suggest", "example", "help", "how to"]):
            state["current_task"] = "suggestions"
        elif any(keyword in user_input for keyword in ["remember", "recall", "past", "history"]):
            state["current_task"] = "memory_search"
        else:
            state["current_task"] = "general"
            
        state["messages"].append(AIMessage(content=f"Intent classified as: {state['current_task']}"))
        return state

    def route_based_on_intent(self, state: AssistantState) -> str:
        """Route to appropriate node based on classified intent."""
        task = state["current_task"]
        
        if task == "query_conversion":
            return "query_conversion"
        elif task == "pattern_analysis":
            return "pattern_analysis"
        elif task == "suggestions":
            return "suggestions"
        elif task == "memory_search":
            return "memory_search"
        else:
            return "general"

    def memory_search_node(self, state: AssistantState) -> AssistantState:
        """Execute memory search operations."""
        try:
            user_input = state["user_input"]
            result = self.memory_tool.execute(query=user_input)
            state["memory_context"] = str(result) if result else "No relevant memories found."
            state["messages"].append(AIMessage(content=f"Memory search completed: {state['memory_context'][:200]}..."))
        except Exception as e:
            state["error_message"] = f"Memory search error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def query_conversion_node(self, state: AssistantState) -> AssistantState:
        """Execute query conversion operations."""
        try:
            user_input = state["user_input"]
            
            # Extract format if specified
            
            
            # Search for similar queries
            similar_queries = self.memory_tool.execute(query=f"similar queries: {user_input[:50]}")
            state["memory_context"] = str(similar_queries) if similar_queries else ""
            
            # Convert the query
            messages = [
                SystemMessage(content=self.system_prompts["converter"]),
                HumanMessage(content=user_input),
            ]
            
            converted_query = self.finetuned_llm.invoke(messages)
            converted_text = converted_query.content
            state["conversion_result"] = converted_text
            
            # Record the conversion
            self._record_conversion(user_input, converted_text, state["memory_context"])
            
            state["messages"].append(AIMessage(content=f"Query converted: {converted_text}"))
            
        except Exception as e:
            state["error_message"] = f"Query conversion error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def pattern_analysis_node(self, state: AssistantState) -> AssistantState:
        """Execute pattern analysis operations."""
        try:
            # Determine analysis type from user input
            analysis_type = "overall"
            time_period = "month"
            
            user_input = state["user_input"].lower()
            if "accuracy" in user_input:
                analysis_type = "accuracy"
            elif "format" in user_input:
                analysis_type = "formats"
            elif "complex" in user_input:
                analysis_type = "complexity"
            elif "topic" in user_input:
                analysis_type = "topics"
            
            # Search for relevant memories
            memories = self.memory_tool.execute(query=f"{analysis_type} {time_period} patterns trends")
            
            # Generate analysis using LLM
            analysis_prompt = f"""
            Based on query history: {memories}
            Provide analysis for {analysis_type} over {time_period} with specific insights and recommendations.
            """
            
            messages = [
                SystemMessage(content=self.system_prompts["analyzer"]),
                HumanMessage(content=analysis_prompt),
            ]
            
            response = self.llm.invoke(messages)
            
            state["analysis_result"] = response.content
            state["messages"].append(AIMessage(content=f"Analysis completed: {state['analysis_result'][:200]}..."))
            
        except Exception as e:
            state["error_message"] = f"Pattern analysis error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def suggestion_node(self, state: AssistantState) -> AssistantState:
        """Generate suggestions based on user history."""
        try:
            user_input = state["user_input"]
            
            # Determine query type
            query_type = "general"
            if "network" in user_input.lower():
                query_type = "network"
            elif "process" in user_input.lower():
                query_type = "process"
            elif "file" in user_input.lower():
                query_type = "file"
            elif "user" in user_input.lower():
                query_type = "user"
            elif "malware" in user_input.lower():
                query_type = "malware"
            
            # Get relevant history
            past_queries = self.memory_tool.execute(query=f"{query_type} queries examples")
            
            # Generate suggestions
            suggestion_prompt = f"""
            Based on user history: {past_queries}
            Provide 5 personalized {query_type} query suggestions with best practices.
            """
            
            messages = [
                SystemMessage(content=self.system_prompts["pattern_recognition"]),
                HumanMessage(content=suggestion_prompt),
            ]
            
            response = self.llm.invoke(messages)
            
            state["analysis_result"] = response.content
            state["messages"].append(AIMessage(content=f"Suggestions generated for {query_type} queries"))
            
        except Exception as e:
            state["error_message"] = f"Suggestion error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def response_generation_node(self, state: AssistantState) -> AssistantState:
        """Generate final response based on processed information."""
        try:
            task = state["current_task"]
            user_input = state["user_input"]
            
            if task == "query_conversion":
                response = f"✅ Query Conversion Result:\n\n**Input:** {user_input}\n**Converted Query:** `{state['conversion_result']}`"
                if state["memory_context"]:
                    response += f"\n\n💡 **Context from past queries:** Found similar conversion patterns in your history."
            
            elif task == "pattern_analysis":
                response = f"📊 **Pattern Analysis Results:**\n\n{state['analysis_result']}"
            
            elif task == "suggestions":
                response = f"💡 **Query Suggestions:**\n\n{state['analysis_result']}"
            
            elif task == "memory_search":
                response = f"🔍 **Memory Search Results:**\n\n{state['memory_context']}"
            
            else:
                # General response
                general_prompt = f"""
                User input: {user_input}
                Available context: {state.get('memory_context', '')}
                
                Provide a helpful response as a security query assistant.
                """
                
                messages = [
                    SystemMessage(content=self.system_prompts["converter"]),
                    HumanMessage(content=general_prompt),
                ]
                
                llm_response = self.llm.invoke(messages)
                response = llm_response.content
            
            # Record conversation in memory
            self.memory_system.record_conversation(
                user_input=user_input,
                ai_output=response
            )
            
            state["messages"].append(AIMessage(content=response))
            
        except Exception as e:
            state["error_message"] = f"Response generation error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def error_handling_node(self, state: AssistantState) -> AssistantState:
        """Handle errors gracefully."""
        error_response = f"❌ I encountered an error: {state['error_message']}\n\nPlease try rephrasing your request or contact support if the issue persists."
        state["messages"].append(AIMessage(content=error_response))
        return state

    def _record_conversion(self, user_input: str, converted_query: str, context: str):
        """Record conversion for learning."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        conversion_record = f"""
        Time: {current_time}
        Input: {user_input}
        Generated Query: {converted_query}
        Context: {context}
        Status: {'Success' if not converted_query.startswith('Error:') else 'Failed'}
        """
        
        self.memory_system.record_conversation(
            user_input=f"Convert query: {user_input}",
            ai_output=conversion_record
        )

    def process_user_input(self, user_input: str, session_id: str = "default") -> str:
        """Process user input through the LangGraph workflow."""
        try:
            # Initialize state
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
            
            # Run the graph
            config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract the final AI message
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I processed your request, but couldn't generate a response. Please try again."
                
        except Exception as e:
            return f"Error processing your request: {str(e)}"

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, feedback: str = "") -> str:
        """Record user corrections to improve future conversions."""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            correction_record = f"""
            Time: {current_time}
            Correction Type: Query Improvement
            Original Natural Language: {original_nl}
            Original Generated Query: {original_query}
            Corrected Query: {corrected_query}
            User Feedback: {feedback}
            """

            self.memory_system.record_conversation(
                user_input=f"Correction: {original_nl}",
                ai_output=correction_record
            )

            return "✅ Correction recorded! I'll learn from this for future conversions."

        except Exception as e:
            return f"Error recording correction: {str(e)}"

    def get_conversion_summary(self) -> str:
        """Generate a summary of today's query conversions."""
        try:
            today = date.today().strftime("%Y-%m-%d")
            today_conversions = self.memory_tool.execute(query=f"today {today} query conversions")

            if not today_conversions or "No relevant memories found" in str(today_conversions):
                return "No query conversions found for today. Try converting your first security query!"

            summary_prompt = f"""
            Based on today's conversions: {today_conversions}
            
            Provide a concise summary including:
            1. Number of queries converted
            2. Common formats used
            3. Success rate
            4. Key patterns
            5. Suggestions for improvement
            """

            messages = [
                SystemMessage(content=self.system_prompts["analyzer"]),
                HumanMessage(content=summary_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    # Wrapper methods for Streamlit/API compatibility
    def chat_with_memory(self, user_input: str, session_id: str = "api") -> str:
        return self.process_user_input(user_input, session_id=session_id)

    def convert_to_query(self, natural_language: str, context: str = "") -> str:
        try:
            similar = ""
            try:
                similar = self.memory_tool.execute(query=f"similar queries: {natural_language[:50]}")
            except Exception:
                pass
            
            messages = [
                SystemMessage(content=self.system_prompts["converter"]),
                HumanMessage(content=natural_language),
            ]
            
            converted = self.finetuned_llm.invoke(messages)
            converted_text = converted.content
            self._record_conversion(natural_language, converted_text, str(similar) if similar else context)
            return converted_text
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_query_patterns(self, analysis_type: str = "overall", time_period: str = "month") -> str:
        try:
            memories = self.memory_tool.execute(query=f"{analysis_type} {time_period} patterns trends")
            analysis_prompt = f"""
            Based on query history: {memories}
            Provide analysis for {analysis_type} over {time_period} with specific insights and recommendations.
            """
            
            messages = [
                SystemMessage(content=self.system_prompts["analyzer"]),
                HumanMessage(content=analysis_prompt),
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Pattern analysis error: {str(e)}"

    def get_query_suggestions(self, query_type: str = "general") -> str:
        try:
            past_queries = self.memory_tool.execute(query=f"{query_type} queries examples successful conversions")
            user_patterns = self.memory_tool.execute(query=f"user preferences {query_type} query patterns")
            suggestion_prompt = f"""
            Based on the user's history with {query_type} queries:
            Past Queries: {past_queries}
            User Patterns: {user_patterns}

            Please provide personalized suggestions including:
            1) 5 example natural language queries for {query_type} security scenarios
            2) Best practices
            3) Common fields/operators
            4) Tips to improve conversion accuracy
            5) Advanced patterns
            """
            
            messages = [
                SystemMessage(content=self.system_prompts["pattern_recognition"]),
                HumanMessage(content=suggestion_prompt),
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Suggestion error: {str(e)}"

    def memory_search(self, query: str) -> str:
        try:
            result = self.memory_tool.execute(query=query)
            return str(result) if result else "No relevant memories found."
        except Exception as e:
            return f"Memory search error: {str(e)}"
