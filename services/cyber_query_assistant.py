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
        logger.info("🚀 Starting CyberQueryAssistant initialization...")
        
        try:
            self.settings = get_settings()
            logger.info("✅ Settings loaded successfully")
            
            # Database configuration
            self.database_user = self.settings.DATABASE_USER
            self.database_password = self.settings.DATABASE_PASSWORD
            self.database_host = self.settings.DATABASE_HOST
            self.database_port = self.settings.DATABASE_PORT
            self.database_name = self.settings.DATABASE_NAME
            
            logger.info(f"🔧 Database config: {self.database_user}@{self.database_host}:{self.database_port}/{self.database_name}")
            
            # Initialize Memori
            logger.info("🧠 Initializing Memori with PostgreSQL database...")
            db_url = f"postgresql+psycopg2://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
            self.memory_system = Memori(
                database_connect=db_url,
                conscious_ingest=True,
                auto_ingest=True,
            )
            logger.info("✅ Memori initialized successfully")

            logger.info("📝 Enabling memory tracking...")
            self.memory_system.enable()
            logger.info("✅ Memory tracking enabled")
            
            # Initialize LLMs
            logger.info("🤖 Initializing LLMs...")
            self.llm = ChatOpenAI(
                model=self.settings.LLM_MODEL, 
                temperature=self.settings.LLM_TEMPERATURE
            )
            logger.info(f"✅ Main LLM initialized: {self.settings.LLM_MODEL}")

            self.finetuned_llm = ChatOpenAI(
                model=self.settings.FINETUNED_MODEL_NAME,
                api_key=self.settings.OPENAI_API_KEY,
                temperature=self.settings.LLM_TEMPERATURE,
                timeout=60,
            )
            logger.info(f"✅ Fine-tuned LLM initialized: {self.settings.FINETUNED_MODEL_NAME}")

            # Create memory tool
            logger.info("🛠️  Creating memory tool...")
            self.memory_tool = create_memory_tool(self.memory_system)
            logger.info("✅ Memory tool created successfully")

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
            logger.info("📋 System prompts configured")

            # Initialize tools
            logger.info("🔧 Creating tools...")
            self.tools = self._create_tools()
            logger.info(f"✅ {len(self.tools)} tools created successfully")
            
            self.memory_saver = MemorySaver()
            logger.info("💾 Memory saver initialized")
     
            # Create the graph
            logger.info("📊 Creating LangGraph workflow...")
            self.graph = self._create_graph()
            logger.info("✅ LangGraph workflow created and compiled")
            
            logger.info("🎉 CyberQueryAssistant initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CyberQueryAssistant: {str(e)}", exc_info=True)
            raise

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the assistant."""
        logger.debug("🛠️  Creating individual tools...")
        
        def memory_search_tool(query: str) -> str:
            """Search memories for relevant information."""
            logger.info(f"🔍 Memory search requested: '{query[:100]}...'")
            try:
                start_time = datetime.now()
                result = self.memory_tool.execute(query=query)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if result:
                    logger.info(f"✅ Memory search completed in {duration:.2f}s, found {len(str(result))} chars")
                    logger.debug(f"Memory search result preview: {str(result)[:200]}...")
                    return str(result)
                else:
                    logger.warning(f"⚠️  Memory search completed in {duration:.2f}s but no results found")
                    return "No relevant memories found."
            except Exception as e:
                logger.error(f"❌ Memory search error for query '{query}': {str(e)}", exc_info=True)
                return f"Memory search error: {str(e)}"

        def convert_to_query_tool(natural_language: str, context: str = "") -> str:
            """Convert natural language to structured query."""
            logger.info(f"🔄 Query conversion requested: '{natural_language[:100]}...'")
            try:
                start_time = datetime.now()
                
                # Search for similar past queries for context
                logger.debug("🔍 Searching for similar past queries...")
                similar_queries = memory_search_tool(f"similar queries: {natural_language[:50]}")
                
                # Call fine-tuned model
                logger.debug("🤖 Calling fine-tuned model for conversion...")
                messages = [
                    SystemMessage(content=self.system_prompts["converter"]),
                    HumanMessage(content=natural_language),
                ]
                
                converted_query = self.finetuned_llm.invoke(messages)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"✅ Query conversion completed in {duration:.2f}s")
                logger.debug(f"Converted query: {converted_query.content[:200]}...")
                
                # Store the conversion in memory
                logger.debug("💾 Recording conversion in memory...")
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
                logger.debug("✅ Conversion recorded in memory")
                
                return converted_query.content
                
            except Exception as e:
                logger.error(f"❌ Query conversion error for input '{natural_language}': {str(e)}", exc_info=True)
                return f"Query conversion error: {str(e)}"

        def analyze_patterns_tool(analysis_type: str, time_period: str = "month") -> str:
            """Analyze patterns in user's query conversion history."""
            logger.info(f"📊 Pattern analysis requested: type='{analysis_type}', period='{time_period}'")
            try:
                start_time = datetime.now()
                
                search_queries = {
                    "accuracy": f"query conversions corrections accuracy {time_period}",
                    "formats": f"output formats preferences {time_period}",
                    "complexity": f"complex queries simple queries {time_period}",
                    "topics": f"security topics network process file {time_period}",
                    "improvements": f"query improvements suggestions {time_period}",
                    "overall": f"conversion patterns trends {time_period}",
                }

                query = search_queries.get(analysis_type, f"{analysis_type} {time_period}")
                logger.debug(f"🔍 Using search query: '{query}'")
                
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

                logger.debug("🤖 Calling LLM for pattern analysis...")
                messages = [
                    SystemMessage(content=self.system_prompts["analyzer"]),
                    HumanMessage(content=analysis_prompt),
                ]

                response = self.llm.invoke(messages)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"✅ Pattern analysis completed in {duration:.2f}s")
                logger.debug(f"Analysis result preview: {response.content[:200]}...")
                
                return response.content

            except Exception as e:
                logger.error(f"❌ Pattern analysis error for type '{analysis_type}': {str(e)}", exc_info=True)
                return f"Pattern analysis error: {str(e)}"

        def get_suggestions_tool(query_type: str) -> str:
            """Generate personalized query suggestions."""
            logger.info(f"💡 Query suggestions requested for type: '{query_type}'")
            try:
                start_time = datetime.now()
                
                logger.debug("🔍 Searching for past queries and user patterns...")
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

                logger.debug("🤖 Calling LLM for suggestion generation...")
                messages = [
                    SystemMessage(content=self.system_prompts["pattern_recognition"]),
                    HumanMessage(content=suggestion_prompt),
                ]

                response = self.llm.invoke(messages)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"✅ Query suggestions generated in {duration:.2f}s")
                logger.debug(f"Suggestions preview: {response.content[:200]}...")
                
                return response.content

            except Exception as e:
                logger.error(f"❌ Suggestion error for query type '{query_type}': {str(e)}", exc_info=True)
                return f"Suggestion error: {str(e)}"

        tools = [
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
        
        logger.debug(f"✅ Created {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        logger.debug("📊 Building LangGraph workflow...")
        
        # Initialize the graph
        workflow = StateGraph(AssistantState)
        logger.debug("📋 StateGraph initialized")

        # Add nodes
        nodes = [
            ("classify_intent", self.classify_intent_node),
            ("memory_search_node", self.memory_search_node),
            ("query_conversion_node", self.query_conversion_node),
            ("pattern_analysis_node", self.pattern_analysis_node),
            ("suggestion_node", self.suggestion_node),
            ("response_generation_node", self.response_generation_node),
            ("error_handling_node", self.error_handling_node)
        ]
        
        for node_name, node_func in nodes:
            workflow.add_node(node_name, node_func)
            logger.debug(f"➕ Added node: {node_name}")

        # Define the workflow edges
        logger.debug("🔗 Setting up workflow edges...")
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
        
        # Add terminal edges
        terminal_edges = [
            ("memory_search_node", "response_generation_node"),
            ("query_conversion_node", "response_generation_node"),
            ("pattern_analysis_node", "response_generation_node"),
            ("suggestion_node", "response_generation_node"),
            ("response_generation_node", END),
            ("error_handling_node", END)
        ]
        
        for from_node, to_node in terminal_edges:
            workflow.add_edge(from_node, to_node)
            logger.debug(f"🔗 Added edge: {from_node} -> {to_node}")

        logger.debug("⚙️  Compiling workflow with memory saver...")
        compiled_graph = workflow.compile(checkpointer=self.memory_saver)
        logger.debug("✅ Workflow compiled successfully")
        
        return compiled_graph

    def classify_intent_node(self, state: AssistantState) -> AssistantState:
        """Classify user intent to route to appropriate workflow."""
        user_input = state["user_input"].lower()
        logger.info(f"🔍 Classifying intent for input: '{user_input[:100]}...'")
        
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
            
        logger.info(f"🎯 Intent classified as: {state['current_task']}")
        state["messages"].append(AIMessage(content=f"Intent classified as: {state['current_task']}"))
        return state

    def route_based_on_intent(self, state: AssistantState) -> str:
        """Route to appropriate node based on classified intent."""
        task = state["current_task"]
        logger.debug(f"🚏 Routing to node based on task: {task}")
        
        route_mapping = {
            "query_conversion": "query_conversion",
            "pattern_analysis": "pattern_analysis", 
            "suggestions": "suggestions",
            "memory_search": "memory_search"
        }
        
        route = route_mapping.get(task, "general")
        logger.debug(f"➡️  Routed to: {route}")
        return route

    def memory_search_node(self, state: AssistantState) -> AssistantState:
        """Execute memory search operations."""
        logger.info("🔍 Executing memory search node")
        try:
            user_input = state["user_input"]
            logger.debug(f"Searching memory for: '{user_input}'")
            
            result = self.memory_tool.execute(query=user_input)
            state["memory_context"] = str(result) if result else "No relevant memories found."
            
            logger.info(f"✅ Memory search completed, context length: {len(state['memory_context'])}")
            state["messages"].append(AIMessage(content=f"Memory search completed: {state['memory_context'][:200]}..."))
        except Exception as e:
            logger.error(f"❌ Memory search node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Memory search error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def query_conversion_node(self, state: AssistantState) -> AssistantState:
        """Execute query conversion operations."""
        logger.info("🔄 Executing query conversion node")
        try:
            user_input = state["user_input"]
            logger.debug(f"Converting query: '{user_input}'")
            
            # Search for similar queries
            logger.debug("🔍 Searching for similar queries...")
            similar_queries = self.memory_tool.execute(query=f"similar queries: {user_input[:50]}")
            state["memory_context"] = str(similar_queries) if similar_queries else ""
            
            # Convert the query
            logger.debug("🤖 Invoking fine-tuned model for conversion...")
            messages = [
                SystemMessage(content=self.system_prompts["converter"]),
                HumanMessage(content=user_input),
            ]
            
            converted_query = self.finetuned_llm.invoke(messages)
            converted_text = converted_query.content
            state["conversion_result"] = converted_text
            
            logger.info(f"✅ Query converted successfully, length: {len(converted_text)}")
            logger.debug(f"Conversion result: {converted_text[:200]}...")
            
            # Record the conversion
            logger.debug("💾 Recording conversion...")
            self._record_conversion(user_input, converted_text, state["memory_context"])
            
            state["messages"].append(AIMessage(content=f"Query converted: {converted_text}"))
            
        except Exception as e:
            logger.error(f"❌ Query conversion node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Query conversion error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def pattern_analysis_node(self, state: AssistantState) -> AssistantState:
        """Execute pattern analysis operations."""
        logger.info("📊 Executing pattern analysis node")
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
            
            logger.debug(f"Analysis type: {analysis_type}, time period: {time_period}")
            
            # Search for relevant memories
            logger.debug("🔍 Searching for relevant memories...")
            memories = self.memory_tool.execute(query=f"{analysis_type} {time_period} patterns trends")
            
            # Generate analysis using LLM
            logger.debug("🤖 Generating analysis with LLM...")
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
            logger.info(f"✅ Pattern analysis completed, result length: {len(response.content)}")
            logger.debug(f"Analysis preview: {response.content[:200]}...")
            
            state["messages"].append(AIMessage(content=f"Analysis completed: {state['analysis_result'][:200]}..."))
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Pattern analysis error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def suggestion_node(self, state: AssistantState) -> AssistantState:
        """Generate suggestions based on user history."""
        logger.info("💡 Executing suggestion node")
        try:
            user_input = state["user_input"]
            
            # Determine query type
            query_type = "general"
            type_keywords = {
                "network": "network",
                "process": "process", 
                "file": "file",
                "user": "user",
                "malware": "malware"
            }
            
            for keyword, qtype in type_keywords.items():
                if keyword in user_input.lower():
                    query_type = qtype
                    break
            
            logger.debug(f"Determined query type: {query_type}")
            
            # Get relevant history
            logger.debug("🔍 Searching for relevant query history...")
            past_queries = self.memory_tool.execute(query=f"{query_type} queries examples")
            
            # Generate suggestions
            logger.debug("🤖 Generating suggestions with LLM...")
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
            logger.info(f"✅ Suggestions generated for {query_type}, result length: {len(response.content)}")
            logger.debug(f"Suggestions preview: {response.content[:200]}...")
            
            state["messages"].append(AIMessage(content=f"Suggestions generated for {query_type} queries"))
            
        except Exception as e:
            logger.error(f"❌ Suggestion node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Suggestion error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def response_generation_node(self, state: AssistantState) -> AssistantState:
        """Generate final response based on processed information."""
        logger.info("📝 Executing response generation node")
        try:
            task = state["current_task"]
            user_input = state["user_input"]
            logger.debug(f"Generating response for task: {task}")
            
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
                logger.debug("🤖 Generating general response with LLM...")
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
            
            logger.info(f"✅ Response generated, length: {len(response)}")
            logger.debug(f"Response preview: {response[:200]}...")
            
            # Record conversation in memory
            logger.debug("💾 Recording conversation in memory...")
            self.memory_system.record_conversation(
                user_input=user_input,
                ai_output=response
            )
            
            state["messages"].append(AIMessage(content=response))
            
        except Exception as e:
            logger.error(f"❌ Response generation node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Response generation error: {str(e)}"
            state["current_task"] = "error"
        
        return state

    def error_handling_node(self, state: AssistantState) -> AssistantState:
        """Handle errors gracefully."""
        logger.warning(f"⚠️  Executing error handling node for error: {state.get('error_message', 'Unknown error')}")
        
        error_response = f"❌ I encountered an error: {state['error_message']}\n\nPlease try rephrasing your request or contact support if the issue persists."
        state["messages"].append(AIMessage(content=error_response))
        
        logger.info("✅ Error handled gracefully")
        return state

    def _record_conversion(self, user_input: str, converted_query: str, context: str):
        """Record conversion for learning."""
        logger.debug(f"💾 Recording conversion - Input: '{user_input[:50]}...', Query length: {len(converted_query)}")
        
        try:
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
            
            logger.debug("✅ Conversion recorded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to record conversion: {str(e)}", exc_info=True)

    def process_user_input(self, user_input: str, session_id: str = "default") -> str:
        """Process user input through the LangGraph workflow."""
        logger.info(f"🎯 Processing user input for session '{session_id}': '{user_input[:100]}...'")
        
        try:
            start_time = datetime.now()
            
            # Initialize state
            logger.debug("📋 Initializing state...")
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
            logger.debug("🚀 Invoking LangGraph workflow...")
            config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = self.graph.invoke(initial_state, config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract the final AI message
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
                logger.info(f"✅ Processing completed in {duration:.2f}s, response length: {len(response)}")
                logger.debug(f"Final response preview: {response[:200]}...")
                return response
            else:
                logger.warning("⚠️  No AI messages found in final state")
                return "I processed your request, but couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Error processing user input: {str(e)}", exc_info=True)
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
