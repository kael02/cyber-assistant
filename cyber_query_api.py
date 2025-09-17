from typing import Dict, Any
from datetime import datetime
import asyncio

from config import logger
from services.cyber_query_assistant import CyberQueryAssistant
from models.responses import (
    QueryConversionRequest, QueryConversionResponse,
    PatternAnalysisRequest, PatternAnalysisResponse,
    SuggestionRequest, SuggestionResponse,
    CorrectionRequest, MemorySearchRequest, MemorySearchResponse
)


class CyberQueryAPI:
    """API interface for the Cyber Query Assistant."""

    def __init__(self):
        """Initialize the API with the assistant."""
        logger.info("🚀 Initializing Cyber Query API...")
        self.assistant = CyberQueryAssistant()
        logger.info("✅ Cyber Query API initialized successfully")

    async def convert_query(self, request: QueryConversionRequest) -> QueryConversionResponse:
        """Convert natural language to structured query."""
        logger.info(f"🔄 API: Query conversion request for session '{request['session_id']}'")
        
        start_time = datetime.now()
        
        try:
            converted_query = self.assistant.convert_to_query(
                natural_language=request["natural_language"],
                context=request.get("context", "")
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            response = QueryConversionResponse(
                converted_query=converted_query,
                field_context_used=True,  # Always true with our implementation
                similar_queries_found="similar queries" in converted_query.lower(),
                conversion_time_seconds=duration
            )
            
            logger.info(f"✅ API: Query conversion completed in {duration:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ API: Query conversion error: {str(e)}", exc_info=True)
            raise

    async def analyze_patterns(self, request: PatternAnalysisRequest) -> PatternAnalysisResponse:
        """Analyze query patterns."""
        logger.info(f"📊 API: Pattern analysis request for session '{request['session_id']}'")
        
        start_time = datetime.now()
        
        try:
            analysis_results = self.assistant.analyze_query_patterns(
                analysis_type=request["analysis_type"],
                time_period=request["time_period"]
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract recommendations from analysis (simple parsing)
            recommendations = []
            if "recommend" in analysis_results.lower():
                lines = analysis_results.split('\n')
                for line in lines:
                    if "recommend" in line.lower() or line.strip().startswith('-'):
                        recommendations.append(line.strip())
            
            response = PatternAnalysisResponse(
                analysis_results=analysis_results,
                patterns_found=analysis_results.count('\n') + 1,  # Simple metric
                recommendations=recommendations[:5],  # Limit to 5
                analysis_time_seconds=duration
            )
            
            logger.info(f"✅ API: Pattern analysis completed in {duration:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ API: Pattern analysis error: {str(e)}", exc_info=True)
            raise

    async def get_suggestions(self, request: SuggestionRequest) -> SuggestionResponse:
        """Get query suggestions."""
        logger.info(f"💡 API: Suggestion request for session '{request['session_id']}'")
        
        try:
            suggestions_text = self.assistant.get_query_suggestions(
                query_type=request["query_type"]
            )
            
            # Parse suggestions from the response (simple parsing)
            lines = suggestions_text.split('\n')
            suggestions = []
            best_practices = []
            example_queries = []
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if "example" in line.lower():
                    current_section = "examples"
                elif "practice" in line.lower() or "tip" in line.lower():
                    current_section = "practices"
                elif line.startswith('-') or line.startswith('*'):
                    if current_section == "examples":
                        example_queries.append(line[1:].strip())
                    elif current_section == "practices":
                        best_practices.append(line[1:].strip())
                    else:
                        suggestions.append(line[1:].strip())
            
            response = SuggestionResponse(
                suggestions=suggestions[:5],
                best_practices=best_practices[:5],
                example_queries=example_queries[:5]
            )
            
            logger.info("✅ API: Suggestions generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"❌ API: Suggestion error: {str(e)}", exc_info=True)
            raise

    async def record_correction(self, request: CorrectionRequest) -> Dict[str, str]:
        """Record user correction."""
        logger.info(f"📝 API: Correction recording request for session '{request['session_id']}'")
        
        try:
            result = self.assistant.record_correction(
                original_nl=request["original_natural_language"],
                original_query=request["original_generated_query"],
                corrected_query=request["corrected_query"],
                feedback=request.get("feedback", "")
            )
            
            logger.info("✅ API: Correction recorded successfully")
            return {"status": "success", "message": result}
            
        except Exception as e:
            logger.error(f"❌ API: Correction recording error: {str(e)}", exc_info=True)
            raise

    async def search_memory(self, request: MemorySearchRequest) -> MemorySearchResponse:
        """Search memory for relevant information."""
        logger.info(f"🔍 API: Memory search request for session '{request['session_id']}'")
        
        start_time = datetime.now()
        
        try:
            results = self.assistant.memory_search(request["query"])
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            response = MemorySearchResponse(
                results=results,
                results_found="No relevant memories found" not in results,
                search_time_seconds=duration
            )
            
            logger.info(f"✅ API: Memory search completed in {duration:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ API: Memory search error: {str(e)}", exc_info=True)
            raise

    async def chat(self, user_input: str, session_id: str = "api") -> Dict[str, Any]:
        """General chat interface with memory."""
        logger.info(f"💬 API: Chat request for session '{session_id}'")
        
        start_time = datetime.now()
        
        try:
            response = self.assistant.chat_with_memory(
                user_input=user_input,
                session_id=session_id
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "response": response,
                "session_id": session_id,
                "response_time_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ API: Chat completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ API: Chat error: {str(e)}", exc_info=True)
            raise

    async def get_conversion_summary(self, session_id: str = "api") -> Dict[str, Any]:
        """Get today's conversion summary."""
        logger.info(f"📊 API: Conversion summary request for session '{session_id}'")
        
        try:
            summary = self.assistant.get_conversion_summary()
            
            result = {
                "summary": summary,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ API: Conversion summary generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ API: Conversion summary error: {str(e)}", exc_info=True)
            raise

    def health_check(self) -> Dict[str, str]:
        """API health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }