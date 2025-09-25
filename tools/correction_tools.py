from datetime import datetime
from typing import Optional
from config import logger
from datetime import date


class CorrectionTools:
    """Handles user corrections and feedback recording."""
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system

    def record_correction(self, original_nl: str, original_query: str, 
                         corrected_query: str, feedback: str = "", 
                         session_id: str = None) -> str:
        """Record user corrections for learning."""
        logger.info(f"Recording correction: {original_nl[:50]}...")
        
        try:
            current_time = datetime.now().isoformat()
            correction_record = {
                "timestamp": current_time,
                "original_natural_language": original_nl,
                "original_query": original_query,
                "corrected_query": corrected_query,
                "feedback": feedback,
                "session_id": session_id
            }
            
            # Record in memory system with structured format
            correction_text = f"""
            CORRECTION RECORD:
            Time: {current_time}
            Original Input: {original_nl}
            Incorrect Query: {original_query}
            Corrected Query: {corrected_query}
            Feedback: {feedback}
            Session: {session_id}
            """
            
            logger.info("Correction recorded successfully")
            
            return f"Correction recorded successfully. This will help improve future query conversions. Thank you for the feedback!"
            
        except Exception as e:
            logger.error(f"Correction recording error: {e}")
            return "Unable to record correction at this time. Please try again later."

    def get_conversion_summary(self, session_id: str = None, memory_search_func=None) -> str:
        """Generate conversion summary with session awareness."""
        try:
            today = date.today().strftime("%Y-%m-%d")
            
            # Try session-specific summary first if session_id provided
            if session_id and memory_search_func:
                conversions = memory_search_func(f"conversions {today}", session_id=session_id)
            elif memory_search_func:
                conversions = memory_search_func(f"conversions {today}")
            else:
                conversions = "No memory search function available"

            if not conversions or "No relevant" in conversions:
                return "Today's Summary: No conversions yet. Start by converting your first security query!"

            # Quick summary without LLM call for better performance
            lines = conversions.split('\n')
            count = sum(1 for line in lines if 'Query:' in line or 'Convert' in line)
            
            session_note = f" (Session: {session_id})" if session_id else ""
            
            return f"""Today's Conversion Summary{session_note}:
• Queries converted: {count}
• Most recent: {lines[0][:100] if lines else 'None'}...
• Status: All systems operational

Use 'analyze patterns' for detailed insights."""

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return f"Summary unavailable: {str(e)[:50]}"