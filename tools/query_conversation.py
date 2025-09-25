from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from config import logger


class QueryConversionTool:
    """Handles query conversion logic with structured output support."""
    
    def __init__(self, structured_output_manager, conversation_context_manager, field_context_manager):
        self.structured_output = structured_output_manager
        self.conversation_context = conversation_context_manager
        self.field_context = field_context_manager

    def convert_to_query(
        self, 
        natural_language: str, 
        context: str = "", 
        conversation_history: list = None, 
        session_id: str = None,
        use_streaming: bool = False,
        include_examples: bool = True
    ) -> Dict[str, Any]:
        """Enhanced query conversion with example context integration."""
        logger.info(f"Converting with enhanced context: '{natural_language[:50]}...'")
        
        try:
            start_time = datetime.now()

            # Skip memory search for simple queries
            skip_memory = len(natural_language.split()) < 3
            
            # Build enhanced context
            if not skip_memory and include_examples:
                combined_context = self.field_context.build_combined_context(natural_language[:100])
                field_context = combined_context['combined_context']
            elif not skip_memory:
                field_context = self.field_context.build_field_context(natural_language[:100])
            else:
                field_context = ""
        
            # Build conversation context
            conv_context = self.conversation_context.build_conversation_context(conversation_history or [])

            # Analyze for continuation intent
            previous_context = self.conversation_context.extract_previous_query_context(conversation_history or [])
            continuation_info = self.conversation_context.detect_continuation_intent(natural_language, conversation_history or [])

            # Build system message
            system_content = self._build_system_message(field_context, conv_context)

            # Build user message with continuation context
            user_message_parts = []
            
            if continuation_info['is_continuation'] and previous_context['has_previous_query']:
                user_message_parts.append(f"""CONTINUATION DETECTED:
Combining with previous query using {continuation_info['conjunction_type']} logic.
Previous Query: {previous_context['previous_query'][:200]}
Previous Conditions: {previous_context['previous_conditions']}
New conditions: {continuation_info['new_conditions']}""")
            
            if context:
                user_message_parts.append(f"ADDITIONAL CONTEXT: {context}")
            
            user_message_parts.append(f"CONVERT THIS: {natural_language}")
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content="\n\n".join(user_message_parts))
            ]

            # Use structured output model
            if use_streaming and hasattr(self.structured_output.streaming_query_llm, 'stream'):
                result = self.structured_output.streaming_query_llm.invoke(messages)
                response_content = self.structured_output.format_structured_response(result, is_streaming=True)
            else:
                result = self.structured_output.structured_query_llm.invoke(messages)
                response_content = self.structured_output.format_structured_response(result)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhanced conversion completed in {duration:.2f}s")


            return response_content

        except Exception as e:
            logger.error(f"Enhanced conversion error: {str(e)}")
            return {
                "response": self.structured_output._format_error_response("CONVERSION_FAILED", 
                                             "Conversion failed due to system error. Please try a simpler query format."),
                "metadata": {
                    "from_timestamp": None,
                    "to_timestamp": None,
                    "query_type": None,
                    "confidence": None
                }
            }

    def _build_system_message(self, field_context: str, conv_context: str) -> str:
        """Build the system message for query conversion."""
        system_content = f"""You are an expert security query translator with advanced conversational context awareness and example-based learning.

          CORE RESPONSIBILITY: Convert natural language to structured query like Graylog format OR identify out-of-scope requests.

          RELEVANCE SCOPE:
          ✅ IN SCOPE: security logs, network analysis, malware detection, user authentication, file monitoring, 
          process analysis, threat hunting, incident response, system monitoring, vulnerability scanning
          ❌ OUT OF SCOPE: weather, cooking, personal life, general knowledge, math problems, entertainment, 
          non-security topics

          CONVERSION GUIDELINES:
          1. For relevant security queries: Return QueryConversionSuccess with proper structured query. This shouldn't include any time constraints. The time constraints is added later.
          2. For out-of-scope requests: Return QueryConversionError with OUT_OF_SCOPE error
          3. Use available field and example context to ensure accurate mapping and syntax
          4. Learn from provided examples - match similar patterns and syntax styles
          5. If query cannot be constructed with available context, return QueryConversionError with NOT_SUPPORTED error
          6. Handle conversational continuations by combining with previous query context
          7. Calculate epoch timestamps when explicit time constraints are mentioned

          EXAMPLE USAGE STRATEGY:
          - Study the provided examples for syntax patterns and field usage
          - Adapt example patterns to match the current query requirements
          - Maintain consistency with demonstrated query structures
          - Use examples as templates while adapting for specific user needs

          CURRENT TIME: {int(datetime.now().timestamp())}

          CONTEXT INFORMATION:
          {field_context if field_context else 'Use common security log fields and standard query patterns'}
        """
        
        if conv_context:
            system_content += f"\n\nRECENT CONVERSATION:\n{conv_context}"
            
        return system_content

    def _is_error_response(self, response: str) -> bool:
        """Check if response indicates an error condition."""
        error_indicators = ['Error:', 'OUT_OF_SCOPE:', 'I\'m specialized in', 'not relevant to security']
        return any(indicator in response for indicator in error_indicators)




