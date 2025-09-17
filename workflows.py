from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from config import logger
from models import AssistantState


class CyberQueryWorkflow:
    """Workflow manager for the Cyber Query Assistant."""

    def __init__(self, tools_manager, memory_system, system_prompts, llm):
        self.tools_manager = tools_manager
        self.memory_system = memory_system
        self.system_prompts = system_prompts
        self.llm = llm

    def create_graph(self, memory_saver) -> StateGraph:
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

        logger.debug("⚙️ Compiling workflow with memory saver...")
        compiled_graph = workflow.compile(checkpointer=memory_saver)
        logger.debug("✅ Workflow compiled successfully")

        return compiled_graph

    def classify_intent_node(self, state: "AssistantState") -> "AssistantState":
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

    def route_based_on_intent(self, state: "AssistantState") -> str:
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
        logger.debug(f"➡️ Routed to: {route}")
        return route

    def memory_search_node(self, state: "AssistantState") -> "AssistantState":
        """Execute memory search operations."""
        logger.info("🔍 Executing memory search node")
        try:
            user_input = state["user_input"]
            logger.debug(f"Searching memory for: '{user_input}'")

            result = self.tools_manager.memory_search_tool(user_input)
            state["memory_context"] = result

            logger.info(f"✅ Memory search completed, context length: {len(state['memory_context'])}")
            state["messages"].append(AIMessage(content=f"Memory search completed: {state['memory_context'][:200]}..."))
        except Exception as e:
            logger.error(f"❌ Memory search node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Memory search error: {str(e)}"
            state["current_task"] = "error"

        return state

    def query_conversion_node(self, state: "AssistantState") -> "AssistantState":
        """Execute query conversion operations."""
        logger.info("🔄 Executing query conversion node")
        try:
            user_input = state["user_input"]
            logger.debug(f"Converting query: '{user_input}'")

            converted_query = self.tools_manager.convert_to_query_tool(user_input)
            state["conversion_result"] = converted_query

            logger.info(f"✅ Query converted successfully, length: {len(converted_query)}")
            logger.debug(f"Conversion result: {converted_query[:200]}...")

            state["messages"].append(AIMessage(content=f"Query converted: {converted_query}"))

        except Exception as e:
            logger.error(f"❌ Query conversion node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Query conversion error: {str(e)}"
            state["current_task"] = "error"

        return state

    def pattern_analysis_node(self, state: "AssistantState") -> "AssistantState":
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

            result = self.tools_manager.analyze_patterns_tool(analysis_type, time_period)
            state["analysis_result"] = result

            logger.info(f"✅ Pattern analysis completed, result length: {len(result)}")
            logger.debug(f"Analysis preview: {result[:200]}...")

            state["messages"].append(AIMessage(content=f"Analysis completed: {state['analysis_result'][:200]}..."))

        except Exception as e:
            logger.error(f"❌ Pattern analysis node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Pattern analysis error: {str(e)}"
            state["current_task"] = "error"

        return state

    def suggestion_node(self, state: "AssistantState") -> "AssistantState":
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

            result = self.tools_manager.get_suggestions_tool(query_type)
            state["analysis_result"] = result

            logger.info(f"✅ Suggestions generated for {query_type}, result length: {len(result)}")
            logger.debug(f"Suggestions preview: {result[:200]}...")

            state["messages"].append(AIMessage(content=f"Suggestions generated for {query_type} queries"))

        except Exception as e:
            logger.error(f"❌ Suggestion node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Suggestion error: {str(e)}"
            state["current_task"] = "error"

        return state

    def response_generation_node(self, state: "AssistantState") -> "AssistantState":
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
                    SystemMessage(content=self.system_prompts.converter),
                    HumanMessage(content=general_prompt),
                ]

                llm_response = self.llm.invoke(messages)
                response = llm_response.content

            logger.info(f"✅ Response generated, length: {len(response)}")
            logger.debug(f"Response preview: {response[:200]}...")

            # Record conversation in memory
            logger.debug("💾 Recording conversation in memory...")
            self.memory_system.record_conversation(
                user_input=f"[Session: {state['session_id']}] {user_input}",
                ai_output=f"[Session: {state['session_id']}] {response}"
            )

            state["messages"].append(AIMessage(content=response))

        except Exception as e:
            logger.error(f"❌ Response generation node error: {str(e)}", exc_info=True)
            state["error_message"] = f"Response generation error: {str(e)}"
            state["current_task"] = "error"

        return state

    def error_handling_node(self, state: "AssistantState") -> "AssistantState":
        """Handle errors gracefully."""
        logger.warning(f"⚠️ Executing error handling node for error: {state.get('error_message', 'Unknown error')}")

        error_response = f"❌ I encountered an error: {state['error_message']}\n\nPlease try rephrasing your request or contact support if the issue persists."
        state["messages"].append(AIMessage(content=error_response))

        logger.info("✅ Error handled gracefully")
        return state