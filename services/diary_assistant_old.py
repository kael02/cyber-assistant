import json
import os
from datetime import date, datetime
from typing import List, Optional, Dict

import litellm
from dotenv import load_dotenv

from memori import Memori, create_memory_tool

load_dotenv()  # Load environment variables from .env file


class PersonalDiaryAssistant:
    """Natural Language to Security Query Converter with advanced memory and learning capabilities."""

    def __init__(self, database_path: str = "security_queries.db"):
        """Initialize the Security Query Assistant."""
        self.database_path = database_path
        
        print("Initializing Memori with PostgreSQL database...")
        self.database_user = os.getenv("DATABASE_USER")
        self.database_password = os.getenv("DATABASE_PASSWORD")
        self.database_host = os.getenv("DATABASE_HOST")
        self.database_port = os.getenv("DATABASE_PORT")
        self.database_name = os.getenv("DATABASE_NAME")
        self.memory_system = Memori(
            database_connect=f"postgresql+psycopg2://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}",
            conscious_ingest=True,
            auto_ingest=True,
            # verbose=True,
        )

        print("Enabling memory tracking...")
        self.memory_system.enable()

        # Create memory tool
        self.memory_tool = create_memory_tool(self.memory_system)

        # Initialize conversation history for session
        self.conversation_history = []

        # System prompts for different functionalities
        self.system_prompts = {
            "converter": """You are an expert security analyst and query translator with advanced memory capabilities.
            Your role is to help users convert natural language security queries into structured application queries
            like graylog format. Use memory_search to understand the user's query patterns,
            preferences, and past conversions. Be precise, learn from corrections, and provide accurate translations.""",
            
            "analyzer": """You are a security query optimization expert. Analyze the user's query patterns,
            conversion accuracy, and usage trends to provide insights. Use memory_search to gather comprehensive
            information about the user's query history. Provide specific recommendations for improving query
            accuracy and efficiency.""",
            
            "pattern_recognition": """You are a security pattern recognition specialist. Help users identify
            common query patterns, optimize their natural language inputs, and learn from their query history.
            Use memory_search to understand their past queries and suggest improvements.""",
        }

        # Available tools for LLM function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search memories for relevant information about past queries, patterns, corrections, and user preferences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories (e.g., 'network queries', 'splunk format preferences', 'correction patterns')",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_to_query",
                    "description": "Convert natural language to structured security query using fine-tuned model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "natural_language": {
                                "type": "string",
                                "description": "Natural language security query to convert",
                            },
                            "output_format": {
                                "type": "string",
                                "description": "Desired output format for the query",
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context or constraints for the conversion",
                            },
                        },
                        "required": ["natural_language"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_query_patterns",
                    "description": "Analyze patterns in user's query conversion history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "enum": [
                                    "accuracy",
                                    "formats",
                                    "complexity",
                                    "topics",
                                    "improvements",
                                    "overall",
                                ],
                                "description": "Type of analysis to perform",
                            },
                            "time_period": {
                                "type": "string",
                                "enum": [
                                    "day",
                                    "week",
                                    "month", 
                                    "quarter",
                                    "all_time",
                                ],
                                "description": "Time period for analysis",
                            },
                        },
                        "required": ["analysis_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_query_suggestions",
                    "description": "Generate personalized query suggestions based on user's history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": [
                                    "network",
                                    "process",
                                    "file",
                                    "user",
                                    "malware",
                                    "general",
                                ],
                                "description": "Type of security query to get suggestions for",
                            },
                        },
                        "required": ["query_type"],
                    },
                },
            },
        ]

        # Fine-tuned model configuration
        self.model_config = {
            "endpoint": os.getenv("FINETUNED_MODEL_ENDPOINT", "http://localhost:8000/convert"),
            "api_key": os.getenv("FINETUNED_MODEL_KEY", "ft:gpt-4.1-mini-2025-04-14:personal:gpt-mini-generator:CC22K4Xm"),
            "timeout": 30
        }

    def memory_search(self, query: str) -> str:
        """Search memories for relevant information."""
        try:
            result = self.memory_tool.execute(query=query)
            return str(result) if result else "No relevant memories found."
        except Exception as e:
            return f"Memory search error: {str(e)}"

    def convert_to_query(self, natural_language: str, output_format: str = "custom", context: str = "") -> str:
        """Convert natural language to structured query using fine-tuned model."""
        try:
            # Search for similar past queries for context
            similar_queries = self.memory_search(f"similar queries: {natural_language[:50]}")
            
            # Call fine-tuned model with actual API call
            converted_query = self._call_finetuned_model(natural_language, output_format, context)
            
            # Check if the conversion was successful
            if converted_query.startswith("Model API error") or converted_query.startswith("Network error") or converted_query.startswith("Model call failed"):
                # Log the error but still store the attempt
                error_msg = converted_query
                converted_query = f"Error: {error_msg}"
            
            # Store the conversion in memory with additional metadata
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            conversion_record = f"""
            Time: {current_time}
            Input: {natural_language}
            Output Format: {output_format}
            Generated Query: {converted_query}
            Context: {context}
            Model Endpoint: {self.model_config['endpoint']}
            Model Version: {self.model_config.get('version', 'unknown')}
            Similar Past Queries: {similar_queries}
            Status: {'Success' if not converted_query.startswith('Error:') else 'Failed'}
            """
            
            self.memory_system.record_conversation(
                user_input=f"Convert query: {natural_language}",
                ai_output=conversion_record
            )
            
            return converted_query
            
        except Exception as e:
            error_result = f"Query conversion error: {str(e)}"
            # Still record the failed attempt for learning
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            error_record = f"""
            Time: {current_time}
            Input: {natural_language}
            Output Format: {output_format}
            Error: {str(e)}
            Status: Failed
            """
            
            self.memory_system.record_conversation(
                user_input=f"Convert query (failed): {natural_language}",
                ai_output=error_record
            )
            
            return error_result

    def _call_finetuned_model(self, natural_language: str, output_format: str, context: str) -> str:
        """Call the fine-tuned model for query conversion."""
        try:
            # This would call your actual fine-tuned model
            # For demo purposes, using rule-based examples
            
            text_lower = natural_language.lower()
            
            if "source ip" in text_lower and "10.168.168.103" in text_lower:
                if output_format == "custom":
                    return 'pname:Radware*'
                elif output_format == "splunk":
                    return 'src_ip="10.168.168.103" process="Radware*"'
                elif output_format == "elastic":
                    return 'source.ip:"10.168.168.103" AND process.name:Radware*'
            
            elif "network traffic" in text_lower:
                if output_format == "custom":
                    return 'event_type:network*'
                elif output_format == "splunk":
                    return 'eventtype="network*"'
            
            # Fallback - in real implementation this would be your model
            return f"converted_query_{output_format}"
            
        except Exception as e:
            return f"Model call failed: {str(e)}"

    def analyze_query_patterns(self, analysis_type: str, time_period: str = "month") -> str:
        """Analyze patterns in user's query conversion history."""
        try:
            # Search for relevant conversion history
            search_queries = {
                "accuracy": f"query conversions corrections accuracy {time_period}",
                "formats": f"output formats preferences {time_period}",
                "complexity": f"complex queries simple queries {time_period}",
                "topics": f"security topics network process file {time_period}",
                "improvements": f"query improvements suggestions {time_period}",
                "overall": f"conversion patterns trends {time_period}",
            }

            query = search_queries.get(analysis_type, f"{analysis_type} {time_period}")
            memories = self.memory_search(query)

            # Use LLM to analyze the patterns
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

            Be specific and provide concrete examples from the conversion history.
            """

            response = litellm.completion(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": self.system_prompts["analyzer"]},
                    {"role": "user", "content": analysis_prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Pattern analysis error: {str(e)}"

    def get_query_suggestions(self, query_type: str) -> str:
        """Generate personalized query suggestions based on user's history."""
        try:
            # Search for relevant past queries of this type
            past_queries = self.memory_search(f"{query_type} queries examples successful conversions")
            user_patterns = self.memory_search(f"user preferences {query_type} query patterns")

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

            Make suggestions specific to this user's query history and preferences.
            """

            response = litellm.completion(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": self.system_prompts["pattern_recognition"]},
                    {"role": "user", "content": suggestion_prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Suggestion error: {str(e)}"

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, 
                         output_format: str, feedback: str = "") -> str:
        """Record user corrections to improve future conversions."""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            correction_record = f"""
            Time: {current_time}
            Correction Type: Query Improvement
            Original Natural Language: {original_nl}
            Original Generated Query: {original_query}
            Corrected Query: {corrected_query}
            Output Format: {output_format}
            User Feedback: {feedback}
            """

            self.memory_system.record_conversation(
                user_input=f"Correction: {original_nl}",
                ai_output=correction_record
            )

            return "✅ Correction recorded! I'll learn from this for future conversions."

        except Exception as e:
            return f"Error recording correction: {str(e)}"

    def chat_with_memory(self, user_input: str, mode: str = "converter") -> str:
        """Process user input with memory-enhanced conversation."""
        try:
            # Initialize conversation if empty
            if not self.conversation_history:
                self.conversation_history = [
                    {
                        "role": "system",
                        "content": self.system_prompts.get(
                            mode, self.system_prompts["converter"]
                        ),
                    }
                ]

            # Add user message to conversation
            self.conversation_history.append({"role": "user", "content": user_input})

            # Make LLM call with function calling
            response = litellm.completion(
                model=self.model_config['api_key'],
                messages=self.conversation_history,
                tools=self.tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response.choices[0].message.tool_calls

            # Handle function calls
            if tool_calls:
                self.conversation_history.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute the appropriate function
                    if function_name == "memory_search":
                        query = function_args.get("query", "")
                        function_response = self.memory_search(query)
                    elif function_name == "convert_to_query":
                        natural_language = function_args.get("natural_language", "")
                        output_format = function_args.get("output_format", "custom")
                        context = function_args.get("context", "")
                        function_response = self.convert_to_query(natural_language, output_format, context)
                    elif function_name == "analyze_query_patterns":
                        analysis_type = function_args.get("analysis_type", "overall")
                        time_period = function_args.get("time_period", "month")
                        function_response = self.analyze_query_patterns(analysis_type, time_period)
                    elif function_name == "get_query_suggestions":
                        query_type = function_args.get("query_type", "general")
                        function_response = self.get_query_suggestions(query_type)
                    else:
                        function_response = "Unknown function called"

                    # Add function result to conversation
                    self.conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": function_response,
                        }
                    )

                # Get final response after function calls
                final_response = litellm.completion(
                    model=self.model_config['api_key'], messages=self.conversation_history
                )

                final_content = final_response.choices[0].message.content
                self.conversation_history.append(
                    {"role": "assistant", "content": final_content}
                )

                # Record this conversation in memory
                self.memory_system.record_conversation(
                    user_input=user_input, ai_output=final_content
                )

                return final_content

            else:
                # No function calls, just respond normally
                content = response_message.content
                self.conversation_history.append(
                    {"role": "assistant", "content": content}
                )

                # Record this conversation in memory
                self.memory_system.record_conversation(
                    user_input=user_input, ai_output=content
                )

                return content

        except Exception as e:
            return f"Error processing your request: {str(e)}"

    def get_conversion_summary(self) -> str:
        """Generate a summary of today's query conversions and insights."""
        try:
            today = date.today().strftime("%Y-%m-%d")
            today_conversions = self.memory_search(
                f"today {today} query conversions accuracy patterns"
            )

            if "No relevant memories found" in today_conversions:
                return "No query conversions found for today. Try converting your first security query!"

            summary_prompt = f"""
            Based on today's query conversions and activities:

            {today_conversions}

            Please provide a helpful summary including:
            1. Number of queries converted today
            2. Most common output formats used
            3. Accuracy and success rate observations
            4. Common query types processed
            5. Any patterns or improvements noticed
            6. Suggestions for tomorrow's query work

            Keep it concise but insightful.
            """

            response = litellm.completion(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": self.system_prompts["analyzer"]},
                    {"role": "user", "content": summary_prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating conversion summary: {str(e)}"

    def clear_session_history(self):
        """Clear the current session's conversation history."""
        self.conversation_history = []


def main():
    """Main function for command-line interface."""
    print("🔍 Security Query Assistant with Memory")
    print("=" * 50)
    print("Welcome to your intelligent security query converter!")
    print(
        "I can help you convert natural language to structured queries and learn from your patterns."
    )
    print("Type 'exit' or 'quit' to stop.\n")

    # Initialize the assistant
    assistant = PersonalDiaryAssistant("security_queries.db")

    print("💡 Try these commands:")
    print("- Convert: 'Find network activity from IP 192.168.1.100'")
    print("- Ask for query suggestions: 'Show me examples of network queries'")
    print("- Analyze patterns: 'What are my most common query types?'")
    print("- Record corrections: 'The query should be process_name:chrome.exe'")
    print("- Type 'summary' for today's conversion overview")
    print("- Specify format: 'Convert to Splunk format: user login from admin account'\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print(
                    "\nAssistant: Goodbye! Your query patterns are safely stored for our next session. 🔍"
                )
                break

            if user_input.lower() == "summary":
                response = assistant.get_conversion_summary()
                print(f"\n📊 Conversion Summary:\n{response}\n")
                continue

            if not user_input:
                continue

            # Process the input
            response = assistant.chat_with_memory(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye! Your query patterns are safely stored. 🔍")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()