from datetime import datetime, date
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool

from config import logger


class CyberQueryTools:
    """Tools manager for the Cyber Query Assistant."""

    def __init__(self, memory_tool, memory_system, field_store, field_context_k, llm, finetuned_llm, system_prompts):
        self.memory_tool = memory_tool
        self.memory_system = memory_system
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.llm = llm
        self.finetuned_llm = finetuned_llm
        self.system_prompts = system_prompts

    def _build_field_context(self, nl_query: str, k: Optional[int] = None) -> str:
        """Search the application's field catalog and format top-k hits for LLM context."""
        try:
            k = k or self.field_context_k
            hits = self.field_store.search_fields(nl_query, limit=k) or []
            if not hits:
                return ""
            
            lines = []
            for h in hits:
                alias = (h.get("alias") or "").strip()
                long_name = (h.get("long_name") or "")[:140].strip()
                dtype = (h.get("data_type") or "").strip()
                definition = (h.get("definition") or "")[:180].strip()
                example = (h.get("example_value") or "")[:80].strip()
                
                lines.append(
                    f"- alias={alias} | type={dtype} | long_name={long_name} | def={definition}"
                    + (f" | example={example}" if example else "")
                )
            
            return "Application field catalog (top matches):\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Field context build error: {e}", exc_info=True)
            return ""

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

    def memory_search_tool(self, query: str) -> str:
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
                logger.warning(f"⚠️ Memory search completed in {duration:.2f}s but no results found")
                return "No relevant memories found."
        except Exception as e:
            logger.error(f"❌ Memory search error for query '{query}': {str(e)}", exc_info=True)
            return f"Memory search error: {str(e)}"

    def convert_to_query_tool(self, natural_language: str, context: str = "") -> str:
        """Convert natural language to structured query (with semantic field context)."""
        logger.info(f"🔄 Query conversion requested: '{natural_language[:100]}...'")
        try:
            start_time = datetime.now()

            # Search for similar past queries for context
            logger.debug("🔍 Searching for similar past queries...")
            similar_queries = self.memory_search_tool(f"similar queries: {natural_language[:50]}")

            # Semantic field context from pgvector catalog
            logger.debug("📚 Building field context via semantic search...")
            field_context = self._build_field_context(natural_language)

            # Call fine-tuned model
            logger.debug("🤖 Calling fine-tuned model for conversion...")
            messages = [
                SystemMessage(content=self.system_prompts.converter),
                SystemMessage(content=(
                    """
                    When APP FIELD CONTEXT is provided, prefer those aliases if they match the user's intent.
                    Do not mention the context and return only the query.
                    """
                )),
            ]
            
            if field_context:
                messages.append(HumanMessage(content=f"APP FIELD CONTEXT\n{field_context}"))
            messages.append(HumanMessage(content=natural_language))

            converted_query = self.finetuned_llm.invoke(messages)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"✅ Query conversion completed in {duration:.2f}s")
            logger.debug(f"Converted query: {converted_query.content[:200]}...")

            # Store the conversion in memory
            logger.debug("💾 Recording conversion in memory...")
            conversion_context = (context or "") + (f"\nSimilar Past Queries: {similar_queries}")
            if field_context:
                conversion_context += f"\n\nFieldContext:\n{field_context}"
            
            self._record_conversion(natural_language, converted_query.content, conversion_context)

            return converted_query.content

        except Exception as e:
            logger.error(f"❌ Query conversion error for input '{natural_language}': {str(e)}", exc_info=True)
            return f"Query conversion error: {str(e)}"

    def analyze_patterns_tool(self, analysis_type: str, time_period: str = "month") -> str:
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

            memories = self.memory_search_tool(query)

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
                SystemMessage(content=self.system_prompts.analyzer),
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

    def get_suggestions_tool(self, query_type: str) -> str:
        """Generate personalized query suggestions."""
        logger.info(f"💡 Query suggestions requested for type: '{query_type}'")
        try:
            start_time = datetime.now()

            logger.debug("🔍 Searching for past queries and user patterns...")
            past_queries = self.memory_search_tool(f"{query_type} queries examples successful conversions")
            user_patterns = self.memory_search_tool(f"user preferences {query_type} query patterns")

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
                SystemMessage(content=self.system_prompts.pattern_recognition),
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
            today_conversions = self.memory_search_tool(f"today {today} query conversions")

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
                SystemMessage(content=self.system_prompts.analyzer),
                HumanMessage(content=summary_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def create_tools(self) -> List[Tool]:
        """Create LangChain tools for the assistant."""
        logger.debug("🛠️ Creating individual tools...")

        tools = [
            Tool(
                name="memory_search",
                description="Search memories for relevant information about past queries, patterns, corrections, and user preferences",
                func=self.memory_search_tool
            ),
            Tool(
                name="convert_to_query",
                description="Convert natural language to structured security query",
                func=self.convert_to_query_tool
            ),
            Tool(
                name="analyze_query_patterns",
                description="Analyze patterns in user's query conversion history",
                func=self.analyze_patterns_tool
            ),
            Tool(
                name="get_query_suggestions",
                description="Generate personalized query suggestions based on user's history",
                func=self.get_suggestions_tool
            )
        ]

        logger.debug(f"✅ Created {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools