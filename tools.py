from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import re
from datetime import datetime
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from config import logger


# Define structured output schemas
class QueryConversionSuccess(BaseModel):
    """Successful security query conversion result."""
    query: str = Field(description="The converted structured security query")
    from_timestamp: Optional[int] = Field(default=None, description="Start time as epoch timestamp")
    to_timestamp: Optional[int] = Field(default=None, description="End time as epoch timestamp")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1)")
    query_type: Optional[str] = Field(default=None, description="Type of security query (network, process, file, etc.)")

class QueryConversionError(BaseModel):
    """Query conversion error response."""
    error: str = Field(description="Error type: OUT_OF_SCOPE, INVALID_INPUT, or CONVERSION_FAILED")
    message: str = Field(description="Human-readable error message")
    suggestions: Optional[List[str]] = Field(default=None, description="Suggested alternatives")

class QueryConversionResult(BaseModel):
    """Union type for query conversion results."""
    result: Union[QueryConversionSuccess, QueryConversionError] = Field(description="Either success or error result")

# Alternative TypedDict versions for streaming support
class QueryConversionSuccessDict(TypedDict):
    """Successful security query conversion result (TypedDict version)."""
    query: Annotated[str, ..., "The converted structured security query"]
    from_timestamp: Annotated[Optional[int], None, "Start time as epoch timestamp"]
    to_timestamp: Annotated[Optional[int], None, "End time as epoch timestamp"]  
    confidence: Annotated[Optional[float], None, "Confidence score (0-1)"]
    query_type: Annotated[Optional[str], None, "Type of security query (network, process, file, etc.)"]

class QueryConversionErrorDict(TypedDict):
    """Query conversion error response (TypedDict version)."""
    error: Annotated[str, ..., "Error type: OUT_OF_SCOPE, INVALID_INPUT, or CONVERSION_FAILED"]
    message: Annotated[str, ..., "Human-readable error message"]
    suggestions: Annotated[Optional[List[str]], None, "Suggested alternatives"]


class CyberQueryTools:
    """Enhanced tools manager designed for workflow-only architecture with structured output."""

    def __init__(self, memory_tool, memory_system, field_store, field_context_k, llm, finetuned_llm, system_prompts):
        self.memory_tool = memory_tool
        self.memory_system = memory_system
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.llm = llm
        self.finetuned_llm = finetuned_llm
        self.system_prompts = system_prompts
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Caching for frequently accessed data
        self._field_cache = {}
        self._cache_lock = threading.Lock()
        
        self.example_context_k = 3
        self.combined_context_enabled = True
        self.example_relevance_threshold = 1.5
        
        # Create structured output models
        self._setup_structured_models()

    def _setup_structured_models(self):
        """Initialize structured output models."""
        try:
            # Create structured LLM for query conversion with Pydantic (preferred for validation)
            self.structured_query_llm = self.finetuned_llm.with_structured_output(
                QueryConversionResult,
                method="function_calling"  # Use function calling if available
            )
            
            # Create streaming version with TypedDict for potential streaming use
            self.streaming_query_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccessDict,
                method="function_calling"
            )
            
            logger.info("Structured output models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize structured models: {e}")
            # Fallback to non-structured approach
            self.structured_query_llm = self.finetuned_llm
            self.streaming_query_llm = self.finetuned_llm

    def _build_conversation_context(self, conversation_history: list, max_messages: int = 4) -> str:
        """Build contextual information from recent conversation history."""
        if not conversation_history:
            return ""
        
        try:
            # Take recent messages (exclude system messages)
            recent_messages = [
                msg for msg in conversation_history[-max_messages:] 
                if hasattr(msg, 'content') and msg.content
            ]
            
            if not recent_messages:
                return ""
            
            context_parts = []
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content[:200]}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Assistant: {msg.content[:200]}")
                elif hasattr(msg, 'type'):
                    if msg.type == 'human':
                        context_parts.append(f"User: {msg.content[:200]}")
                    elif msg.type == 'ai':
                        context_parts.append(f"Assistant: {msg.content[:200]}")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return ""

    def _extract_previous_query_context(self, conversation_history: list) -> Dict[str, Any]:
        """Extract context from previous queries to understand follow-up intent."""
        if not conversation_history:
            return {}
        
        context = {
            'has_previous_query': False,
            'previous_query': '',
            'previous_conditions': [],
            'query_type': '',
            'continuation_indicators': []
        }
        
        try:
            # Look for the most recent AI response containing a query
            for msg in reversed(conversation_history):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    
                    # Check if this contains a generated query
                    if '```' in content or any(keyword in content.lower() for keyword in ['source_ip:', 'destination_ip:', 'SELECT', 'WHERE']):
                        context['has_previous_query'] = True
                        context['previous_query'] = content
                        
                        # Extract conditions from the query
                        context['previous_conditions'] = self._extract_query_conditions(content)
                        break
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return context

    def _extract_query_conditions(self, query_text: str) -> List[str]:
        """Extract conditions from a previously generated query."""
        conditions = []
        
        try:
            # Common patterns in security queries
            patterns = [
                r'source_ip:\s*([^\s\)]+)',
                r'destination_ip:\s*([^\s\)]+)',
                r'port:\s*([^\s\)]+)',
                r'protocol:\s*([^\s\)]+)',
                r'user:\s*([^\s\)]+)',
                r'process:\s*([^\s\)]+)',
                r'file:\s*([^\s\)]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    conditions.append(match.strip('"\''))
            
        except Exception as e:
            logger.error(f"Error extracting conditions: {e}")
            
        return conditions

    def _detect_continuation_intent(self, user_input: str, conversation_history: list) -> Dict[str, Any]:
        """Detect if the user input is a continuation of a previous query."""
        continuation_info = {
            'is_continuation': False,
            'conjunction_type': 'AND',
            'new_conditions': [],
            'modification_type': 'add'
        }
        
        user_lower = user_input.lower().strip()
        
        # Strong continuation indicators
        continuation_patterns = [
            (r'^and\s+(.+)', 'AND', 'add'),
            (r'^also\s+(.+)', 'AND', 'add'),
            (r'^plus\s+(.+)', 'AND', 'add'),
            (r'^with\s+(.+)', 'AND', 'add'),
            (r'^including\s+(.+)', 'AND', 'add'),
            (r'^or\s+(.+)', 'OR', 'add'),
            (r'^but\s+not\s+(.+)', 'NOT', 'exclude'),
            (r'^exclude\s+(.+)', 'NOT', 'exclude'),
            (r'^except\s+(.+)', 'NOT', 'exclude')
        ]
        
        for pattern, conjunction, modification in continuation_patterns:
            match = re.match(pattern, user_lower)
            if match:
                continuation_info['is_continuation'] = True
                continuation_info['conjunction_type'] = conjunction
                continuation_info['modification_type'] = modification
                continuation_info['new_conditions'] = [match.group(1)]
                break
        
        # Check for implicit continuation
        if not continuation_info['is_continuation'] and conversation_history:
            previous_context = self._extract_previous_query_context(conversation_history)
            if (previous_context['has_previous_query'] and 
                self._has_similar_field_references(user_input, previous_context['previous_conditions'])):
                continuation_info['is_continuation'] = True
                continuation_info['new_conditions'] = [user_input]
        
        return continuation_info

    def _has_similar_field_references(self, current_input: str, previous_conditions: List[str]) -> bool:
        """Check if current input references similar fields as previous conditions."""
        current_lower = current_input.lower()
        
        field_patterns = {
            'ip': ['ip', 'address'],
            'port': ['port'],
            'user': ['user', 'username'],
            'process': ['process', 'executable'],
            'file': ['file', 'path'],
            'protocol': ['protocol', 'tcp', 'udp']
        }
        
        for condition in previous_conditions:
            for field_type, keywords in field_patterns.items():
                if any(keyword in condition.lower() for keyword in keywords):
                    if any(keyword in current_lower for keyword in keywords):
                        return True
        
        return False


    @lru_cache(maxsize=100)
    def _build_example_context_cached(self, nl_query: str, k: Optional[int] = None) -> str:
        """Completely field-agnostic approach using progressive query degradation."""
        try:
            k = k or self.example_context_k
            
            # Progressive search strategy - start specific, get broader
            search_strategies = [
                self._extract_all_meaningful_words,      # All meaningful words
                self._extract_action_and_object_words,   # Action + object focus  
                self._extract_domain_words,              # Security domain words
                self._extract_core_concepts,             # Abstract concepts
            ]
            
            all_examples = []
            seen_examples = set()
            
            for i, strategy in enumerate(search_strategies):
                search_query = strategy(nl_query)
                if not search_query:
                    continue
                    
                logger.debug(f"Search attempt {i+1}: '{search_query}'")
                
                examples = self.field_store.search_examples(
                    search_query,
                    limit=k * 2,
                    relevance_threshold=self.example_relevance_threshold
                ) or []
                
                # Deduplicate by query content
                new_examples = []
                for ex in examples:
                    query_hash = hash(ex.get("query", "").strip().lower())
                    if query_hash not in seen_examples:
                        seen_examples.add(query_hash)
                        new_examples.append(ex)
                        all_examples.append(ex)
                
                logger.debug(f"  Found {len(new_examples)} new examples")
                
                # If we have enough good examples, stop early
                if len(all_examples) >= k and i >= 1:
                    break
            
            if not all_examples:
                logger.debug(f"No examples found for: '{nl_query}'")
                return ""
            
            # Rank all examples using content-based scoring
            ranked_examples = self._rank_examples_by_content_similarity(nl_query, all_examples)
            final_examples = ranked_examples[:k]
            
            logger.debug(f"Final selection: {len(final_examples)} examples")
            return self._format_examples_with_relevance_scores(final_examples, nl_query)
            
        except Exception as e:
            logger.error(f"Field-agnostic context error: {e}")
            return ""

    def _extract_all_meaningful_words(self, query: str) -> str:
        """Extract all potentially meaningful words, excluding only common stop words."""
        # Very minimal stop word list - keep most words
        minimal_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words, including numbers and technical terms
        words = re.findall(r'\b\w+\b', query.lower())
        meaningful_words = [w for w in words if w not in minimal_stops and len(w) > 1]
        
        return " ".join(meaningful_words)

    def _extract_action_and_object_words(self, query: str) -> str:
        """Focus on action words and potential object words."""
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Common action words in security queries
        actions = []
        objects = []
        other = []
        
        for word in words:
            if word in {'select', 'find', 'show', 'get', 'search', 'list', 'display', 'filter', 'query'}:
                actions.append(word)
            elif word in {'logs', 'events', 'data', 'records', 'entries', 'traffic', 'connections', 'processes', 'files'}:
                objects.append(word)
            elif len(word) > 2 and word not in {'the', 'and', 'with', 'from', 'that'}:
                other.append(word)
        
        # Prioritize actions and objects, but include other meaningful terms
        result_words = actions + objects + other[:5]  # Limit other words
        return " ".join(result_words)

    def _extract_domain_words(self, query: str) -> str:
        """Extract words that are likely related to security/IT domain."""
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        domain_words = []
        for word in words:
            # Include any word that might be security/network/system related
            # This is intentionally broad to capture domain-specific terminology
            if (len(word) >= 3 and 
                word not in {'the', 'and', 'with', 'from', 'that', 'this', 'are', 'was', 'were', 'have', 'has'}):
                domain_words.append(word)
        
        return " ".join(domain_words)

    def _extract_core_concepts(self, query: str) -> str:
        """Extract the most abstract, core concepts."""
        query_lower = query.lower()
        
        # Look for time-related concepts
        time_indicators = []
        if any(term in query_lower for term in ['time', 'minute', 'hour', 'day', 'last', 'recent', 'past', 'ago']):
            time_indicators.extend(['time', 'temporal'])
        
        # Look for data/query concepts
        data_indicators = []
        if any(term in query_lower for term in ['select', 'find', 'show', 'get', 'search', 'query']):
            data_indicators.extend(['search', 'query'])
        
        if any(term in query_lower for term in ['log', 'event', 'record', 'data']):
            data_indicators.extend(['data', 'logs'])
        
        # Look for network concepts  
        network_indicators = []
        if any(term in query_lower for term in ['network', 'connection', 'traffic', 'address', 'ip', 'port']):
            network_indicators.extend(['network'])
        
        # Combine all concept indicators
        concepts = time_indicators + data_indicators + network_indicators
        
        # If no specific concepts found, use a very general search
        if not concepts:
            concepts = ['security', 'analysis']
        
        return " ".join(concepts)

    def _rank_examples_by_content_similarity(self, nl_query: str, examples: List[Dict]) -> List[Dict]:
        """Rank examples by content similarity without field-specific logic."""
        if not examples:
            return []
        
        nl_words = set(re.findall(r'\b\w+\b', nl_query.lower()))
        nl_length = len(nl_query.split())
        
        scored_examples = []
        
        for ex in examples:
            score = 0.0
            description = ex.get("description", "").lower()
            query = ex.get("query", "").lower() 
            combined_text = f"{description} {query}"
            
            # Base vector similarity score (lower distance = higher relevance)
            distance = ex.get("distance", 1.0)
            base_score = max(0, 1.5 - distance)  # Convert distance to similarity
            score += base_score * 2.0
            
            # Word overlap scoring
            ex_words = set(re.findall(r'\b\w+\b', combined_text))
            overlap = len(nl_words & ex_words)
            overlap_ratio = overlap / len(nl_words) if nl_words else 0
            score += overlap_ratio * 3.0
            
            # Partial word matching (for technical terms)
            partial_matches = 0
            for nl_word in nl_words:
                if len(nl_word) > 3:  # Only for longer words
                    for ex_word in ex_words:
                        if nl_word in ex_word or ex_word in nl_word:
                            partial_matches += 0.5
            score += min(partial_matches, 2.0)  # Cap the bonus
            
            # Length similarity bonus (similar complexity queries)
            ex_length = len(f"{description} {query}".split())
            if ex_length > 0:
                length_ratio = min(nl_length, ex_length) / max(nl_length, ex_length)
                if length_ratio > 0.5:  # Similar length queries
                    score += 0.5
            
            # Query structure similarity (very basic)
            nl_has_numbers = bool(re.search(r'\d', nl_query))
            ex_has_numbers = bool(re.search(r'\d', combined_text))
            if nl_has_numbers == ex_has_numbers:
                score += 0.3
            
            # Prefer examples with actual query syntax
            if ':' in query and len(query) > 10:
                score += 0.4
            
            # Slight penalty for overly long examples
            if len(combined_text) > 300:
                score -= 0.2
                
            scored_examples.append((score, ex))
        
        # Sort by score descending
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored_examples]

    def _format_examples_with_relevance_scores(self, examples: List[Dict], original_query: str) -> str:
        """Format examples showing why they were selected."""
        if not examples:
            return ""
        
        example_lines = []
        original_words = set(re.findall(r'\b\w+\b', original_query.lower()))
        
        for i, ex in enumerate(examples, 1):
            description = ex.get("description", "").strip()
            query = ex.get("query", "").strip()
            distance = ex.get("distance", 1.0)
            
            # Calculate match indicators
            combined_text = f"{description} {query}".lower()
            ex_words = set(re.findall(r'\b\w+\b', combined_text))
            matching_words = original_words & ex_words
            
            # Format the example
            example_text = f"{i}. {description}"
            
            # Add query snippet
            if len(query) <= 100:
                query_display = query
            else:
                query_display = query[:97] + "..."
            example_text += f"\n   Query: {query_display}"
            
            # Show relevance indicators for debugging
            if matching_words and len(matching_words) <= 5:
                match_list = ", ".join(sorted(matching_words))
                example_text += f"\n   Matches: {match_list}"
            
            # Show similarity score
            similarity = max(0, 1.5 - distance)
            if similarity > 0.1:
                example_text += f"\n   Relevance: {similarity:.2f}"
            
            example_lines.append(example_text)
        
        header = f"Found {len(examples)} relevant examples:"
        return f"{header}\n\n" + "\n\n".join(example_lines)

    def _filter_and_rank_examples(self, nl_query: str, examples: List[Dict], limit: int) -> List[Dict]:
        """Smart filtering and ranking of examples based on query characteristics."""
        if not examples:
            return []
        
        nl_lower = nl_query.lower()
        scored_examples = []
        
        for ex in examples:
            score = 0.0
            description = ex.get("description", "").lower()
            query = ex.get("query", "").lower()
            category = ex.get("category", "").lower()
            complexity = ex.get("complexity", "").lower()
            
            # Base relevance score (from vector search distance - lower is better)
            base_score = 1.0 - min(ex.get("distance", 0.5), 1.0)
            score += base_score * 2.0
            
            # Keyword matching boost
            query_keywords = self._extract_security_keywords(nl_query)
            example_keywords = self._extract_security_keywords(f"{description} {query}")
            
            keyword_overlap = len(set(query_keywords) & set(example_keywords))
            if keyword_overlap > 0:
                score += keyword_overlap * 0.3
            
            # Category relevance boost
            if category:
                if any(cat_word in nl_lower for cat_word in category.split()):
                    score += 0.5
            
            # Complexity preference (prefer simpler examples for complex queries)
            query_complexity = self._estimate_query_complexity(nl_query)
            if complexity:
                if query_complexity == "simple" and complexity == "simple":
                    score += 0.3
                elif query_complexity == "complex" and complexity in ["simple", "medium"]:
                    score += 0.2  # Prefer simpler examples for complex queries
            
            # Field usage alignment
            nl_fields = self._extract_field_references(nl_query)
            example_fields = self._extract_field_references(query)
            
            if nl_fields and example_fields:
                field_overlap = len(set(nl_fields) & set(example_fields))
                if field_overlap > 0:
                    score += field_overlap * 0.25
            
            # Penalize overly long examples (they clutter the context)
            if len(query) > 200:
                score -= 0.2
            
            scored_examples.append((score, ex))
        
        # Sort by score (descending) and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:limit]]

    def _extract_security_keywords(self, text: str) -> List[str]:
        """Extract security-relevant keywords for better example matching."""
        keywords = []
        text_lower = text.lower()
        
        # Network keywords
        network_terms = ['ip', 'network', 'connection', 'traffic', 'port', 'protocol', 'dns', 'tcp', 'udp']
        keywords.extend([term for term in network_terms if term in text_lower])
        
        # Security keywords
        security_terms = ['malware', 'threat', 'attack', 'suspicious', 'malicious', 'vulnerability', 'exploit']
        keywords.extend([term for term in security_terms if term in text_lower])
        
        # Event keywords
        event_terms = ['event', 'log', 'audit', 'activity', 'action', 'process', 'file', 'user']
        keywords.extend([term for term in event_terms if term in text_lower])
        
        # Time keywords
        time_terms = ['time', 'date', 'recent', 'last', 'between', 'today', 'yesterday', 'hour', 'day']
        keywords.extend([term for term in time_terms if term in text_lower])
        
        return list(set(keywords))  # Remove duplicates

    def _extract_field_references(self, text: str) -> List[str]:
        """Extract potential field references from natural language text."""
        fields = []
        text_lower = text.lower()
        
        # Common field patterns
        field_patterns = {
            'source_ip': ['source ip', 'src ip', 'from ip', 'originating ip'],
            'destination_ip': ['destination ip', 'dest ip', 'dst ip', 'target ip', 'to ip'],
            'user': ['user', 'username', 'account', 'login'],
            'process': ['process', 'executable', 'program', 'application'],
            'file': ['file', 'path', 'filename', 'document'],
            'port': ['port', 'service'],
            'timestamp': ['time', 'date', 'when', 'timestamp']
        }
        
        for field, patterns in field_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                fields.append(field)
        
        return fields

    def _estimate_query_complexity(self, nl_query: str) -> str:
        """Estimate the complexity of a natural language query."""
        query_lower = nl_query.lower()
        word_count = len(nl_query.split())
        
        # Complex indicators
        complex_terms = ['and', 'or', 'not', 'between', 'range', 'group by', 'aggregate', 'join']
        complex_count = sum(1 for term in complex_terms if term in query_lower)
        
        # Time complexity
        time_terms = ['last', 'between', 'from', 'to', 'during', 'since', 'until']
        has_time_complexity = any(term in query_lower for term in time_terms)
        
        if word_count <= 5 and complex_count == 0:
            return "simple"
        elif word_count <= 10 and complex_count <= 1 and not has_time_complexity:
            return "medium"
        else:
            return "complex"

    def _build_combined_context(self, nl_query: str) -> Dict[str, str]:
        """Build combined field and example context with smart optimization."""
        context = {
            'field_context': '',
            'example_context': '',
            'combined_context': ''
        }
        
        try:
            # Get both contexts
            field_context = self._build_field_context_cached(nl_query[:100])
            example_context = self._build_example_context_cached(nl_query[:100])
            
            context['field_context'] = field_context
            context['example_context'] = example_context
            
            # Create optimized combined context
            if field_context and example_context:
                # Limit field context if we have good examples
                limited_fields = field_context
                if len(field_context) > 300:
                    field_lines = field_context.split('\n')
                    limited_fields = '\n'.join(field_lines[:6])  # Limit to 6 lines
                
                context['combined_context'] = f"{limited_fields}\n\n{example_context}"
            elif field_context:
                context['combined_context'] = field_context
            elif example_context:
                context['combined_context'] = example_context
            
        except Exception as e:
            logger.error(f"Combined context error: {e}")
        
        return context

    @lru_cache(maxsize=100)
    def _build_field_context_cached(self, nl_query: str, k: Optional[int] = None) -> str:
        """Cached version of field context building."""
        try:
            k = k or self.field_context_k
            hits = self.field_store.search_fields(nl_query, limit=k) or []
            if not hits:
                return ""
            
            lines = []
            for h in hits:
                alias = (h.get("alias") or "").strip()
                long_name = (h.get("long_name") or "")[:100].strip()
                dtype = (h.get("data_type") or "").strip()
                definition = (h.get("definition") or "")[:120].strip()
                
                lines.append(f"- {alias} | {dtype} | {long_name}")
                if len(lines) >= 5:
                    break
            
            return "Field context:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Field context error: {e}")
            return ""

    def _record_conversion_async(self, user_input: str, converted_query: str):
        """Async recording to avoid blocking."""
        def record():
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                record = f"Time: {current_time}\nInput: {user_input}\nQuery: {converted_query}"
                # Record in memory system
                logger.debug(f"Recording conversion: {user_input[:50]}...")
            except Exception as e:
                logger.error(f"Recording error: {e}")
        
        self.executor.submit(record)

    def convert_to_query_tool(
        self, 
        natural_language: str, 
        context: str = "", 
        conversation_history: list = None, 
        session_id: str = None,
        use_streaming: bool = False,
        include_examples: bool = True
    ) -> str:
        """Enhanced query conversion with example context integration."""
        logger.info(f"Converting with enhanced context: '{natural_language[:50]}...'")
        
        try:
            start_time = datetime.now()

            # Skip memory search for simple queries
            skip_memory = len(natural_language.split()) < 3
            
            # Build enhanced context with both fields and examples
            if not skip_memory and self.combined_context_enabled and include_examples:
                combined_context = self._build_combined_context(natural_language[:100])
                field_context = combined_context['combined_context']
            elif not skip_memory:
                field_context = self._build_field_context_cached(natural_language[:100])
            else:
                field_context = ""
        
            # Build conversation context
            conv_context = self._build_conversation_context(conversation_history or [])

            # Analyze for continuation intent
            previous_context = self._extract_previous_query_context(conversation_history or [])
            continuation_info = self._detect_continuation_intent(natural_language, conversation_history or [])

            # Enhanced system message with example-aware instructions
            system_content = f"""You are an expert security query translator with advanced conversational context awareness and example-based learning.

CORE RESPONSIBILITY: Convert natural language to structured query format OR identify out-of-scope requests.

RELEVANCE SCOPE:
✅ IN SCOPE: security logs, network analysis, malware detection, user authentication, file monitoring, 
process analysis, threat hunting, incident response, system monitoring, vulnerability scanning
❌ OUT OF SCOPE: weather, cooking, personal life, general knowledge, math problems, entertainment, 
non-security topics

CONVERSION GUIDELINES:
1. For relevant security queries: Return QueryConversionSuccess with proper structured query
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

            # Build user message with continuation context
            user_message_parts = []
            
            # Add continuation analysis if detected
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
            if use_streaming and hasattr(self.streaming_query_llm, 'stream'):
                result = self.streaming_query_llm.invoke(messages)
                response_content = self._format_structured_response(result, is_streaming=True)
            else:
                result = self.structured_query_llm.invoke(messages)
                response_content = self._format_structured_response(result)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhanced conversion completed in {duration:.2f}s")

            # Async recording for successful conversions
            if not self._is_error_response(response_content):
                self._record_conversion_async(natural_language, response_content)

            return response_content

        except Exception as e:
            logger.error(f"Enhanced conversion error: {str(e)}")
            return self._format_error_response("CONVERSION_FAILED", 
                                             "Conversion failed due to system error. Please try a simpler query format.")

    def _format_structured_response(self, result: Union[QueryConversionResult, dict], is_streaming: bool = False) -> str:
        """Format the structured output result into a user-friendly response."""
        try:
            if is_streaming:
                # Handle TypedDict result from streaming
                if isinstance(result, dict):
                    if 'error' in result:
                        return {
                            "response": self._format_error_response(result['error'], result['message'], 
                                                         result.get('suggestions'),),
                            "metadata": {
                                "from_timestamp": None,
                                "to_timestamp": None,
                                "query_type": None,
                                "confidence": None
                            }
                        }
                                                      
                    else:
                        return {
                            "response": result.get('query', 'No query generated'),
                            "metadata": {
                                "from_timestamp": None,
                                "to_timestamp": None,
                                "query_type": None,
                                "confidence": None
                            }
                        }
            else:
                # Handle Pydantic result
                if hasattr(result, 'result'):
                    inner_result = result.result
                    if isinstance(inner_result, QueryConversionSuccess):
                        return {
                            "response": inner_result.query,
                            "metadata": {
                                "from_timestamp": inner_result.from_timestamp,
                                "to_timestamp": inner_result.to_timestamp,
                                "query_type": inner_result.query_type,
                                "confidence": inner_result.confidence
                            }
                        }
                    elif isinstance(inner_result, QueryConversionError):
                        return {
                            "response": self._format_error_response(inner_result.error, inner_result.message, 
                                                         inner_result.suggestions,),
                            "metadata": {
                                "from_timestamp": None,
                                "to_timestamp": None,
                                "query_type": None,
                                "confidence": None
                            }
                        }
                elif isinstance(result, dict):
                    # Fallback for dict-like results
                    return {
                        "response": result.get('query', str(result)),
                        "metadata": {
                            "from_timestamp": None,
                            "to_timestamp": None,
                            "query_type": None,
                            "confidence": None
                        }
                    }
                
            # Fallback - return string representation
            return {
                "response": str(result),
                "metadata": {
                    "from_timestamp": None,
                    "to_timestamp": None,
                    "query_type": None,
                    "confidence": None
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {e}")
            return self._format_error_response("CONVERSION_FAILED", 
                                             "Failed to format query conversion result.")

    def _format_error_response(self, error_type: str, message: str, suggestions: Optional[List[str]] = None) -> str:
        """Format error responses consistently."""
        if error_type == "OUT_OF_SCOPE":
            response = f"{message}"
            if suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
            return response
        elif error_type == "NOT_SUPPORTED":
            response = f"{message}"
            if suggestions:
                response += f"\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions)
            return response
        else:
            return f"Error: {message}"

    def _is_error_response(self, response: str) -> bool:
        """Check if response indicates an error condition."""
        error_indicators = ['Error:', 'OUT_OF_SCOPE:', 'I\'m specialized in', 'not relevant to security']
        return any(indicator in response for indicator in error_indicators)

    # Memory search tool with session awareness
    def memory_search_tool(self, query: str, session_id: str = None) -> str:
        """Enhanced memory search with session awareness."""
        logger.info(f"Memory search: '{query[:50]}...'")
        try:
            start_time = datetime.now()
            
            # If session_id provided, search session-specific first
            if session_id and hasattr(self.memory_system, 'search_session_memories'):
                session_results = self.memory_system.search_session_memories(
                    session_id, query, limit=3
                )
                if session_results:
                    result = self._format_session_search_results(session_results)
                else:
                    result = self.memory_tool.execute(query=query)
            else:
                result = self.memory_tool.execute(query=query)
            
            duration = (datetime.now() - start_time).total_seconds()
            if duration > 3.0:
                logger.warning(f"Slow memory search: {duration:.2f}s")
            
            if result:
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... [truncated]"
                return result_str
            else:
                return "No relevant memories found."
                
        except Exception as e:
            logger.error(f"Memory search error: {str(e)}")
            return "Memory search temporarily unavailable."

    def _format_session_search_results(self, results: List[Dict]) -> str:
        """Format session search results."""
        if not results:
            return "No relevant memories found in this session."
        
        formatted = "Found in this session:\n\n"
        for i, result in enumerate(results[:3], 1):
            timestamp = result.get('timestamp', 'Unknown time')
            content = result.get('content', str(result))[:150]
            formatted += f"{i}. {timestamp}: {content}...\n"
        
        return formatted

    # Rest of the methods remain the same...
    def analyze_patterns_tool(self, analysis_type: str, time_period: str = "week", conversation_history: list = None, session_id: str = None) -> str:
        """Enhanced pattern analysis with conversation context."""
        logger.info(f"Pattern analysis: {analysis_type}")
        
        # Use shorter time periods for faster results
        time_period = "week" if time_period == "month" else time_period
        
        try:
            # Build search query
            search_map = {
                "accuracy": f"conversions accuracy {time_period}",
                "formats": f"query formats {time_period}",
                "complexity": f"query complexity {time_period}",
                "topics": f"security topics {time_period}",
                "overall": f"query patterns {time_period}",
            }

            query = search_map.get(analysis_type, f"{analysis_type} patterns")
            
            # Use session-aware memory search if possible
            if session_id:
                memories = self.memory_search_tool(query, session_id=session_id)
            else:
                memories = self.memory_search_tool(query)

            # Build conversation context
            conv_context = self._build_conversation_context(conversation_history or [])

            # Enhanced analysis prompt with context
            analysis_prompt = f"""
            Analyze these query patterns ({analysis_type}):
            {memories[:800]}
            """
            
            if conv_context:
                analysis_prompt += f"\n\nCurrent conversation context:\n{conv_context[:400]}"
            
            analysis_prompt += "\n\nProvide: 1) Key trends 2) Common formats 3) Quick recommendations"

            messages = [
                SystemMessage(content="Provide concise pattern analysis considering the conversation context."),
                HumanMessage(content=analysis_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return f"Pattern analysis for {analysis_type}: Unable to complete analysis at this time."

    def get_suggestions_tool(self, query_type: str, conversation_history: list = None, session_id: str = None) -> str:
        """Enhanced suggestions with conversation context."""
        logger.info(f"Suggestions for: {query_type}")
        
        # Build conversation context for dynamic suggestions
        conv_context = self._build_conversation_context(conversation_history or [])
        
        # Pre-built suggestions for speed
        quick_suggestions = {
            "network": """Network Query Examples:
• "Show network connections from suspicious IPs"
• "Find processes with unusual network activity"
• "Display DNS queries to malicious domains"
• "Search for lateral movement patterns"
• "Identify data exfiltration attempts"

Best Practices: Use specific IP ranges, timeframes, and protocols.""",
            
            "process": """Process Query Examples:
• "Find processes running from temp directories"
• "Show processes with unsigned executables"
• "Identify processes with high privilege escalation"
• "Search for processes spawning unusual children"
• "Find processes with suspicious command lines"

Best Practices: Include parent-child relationships and execution paths.""",
            
            "file": """File Query Examples:
• "Show files created in system directories"
• "Find files with unusual extensions"
• "Identify recently modified critical files"
• "Search for files with suspicious hashes"
• "Display files accessed by unauthorized users"
"""
        }
        
        # If we have conversation context and it's a known type, enhance the suggestions
        if query_type in quick_suggestions and conv_context:
            try:
                enhancement_prompt = f"""
                Based on this conversation context:
                {conv_context[:300]}
                
                Enhance these suggestions for {query_type} queries with more specific, contextual examples.
                Keep it concise and practical.
                """
                
                messages = [
                    SystemMessage(content="Provide enhanced, contextual security query suggestions."),
                    HumanMessage(content=enhancement_prompt),
                ]
                
                enhanced_response = self.llm.invoke(messages)
                return f"{quick_suggestions[query_type]}\n\nContext-specific suggestions:\n{enhanced_response.content}"
                
            except Exception:
                # Fallback to pre-built suggestions if enhancement fails
                return quick_suggestions[query_type]
        
        elif query_type in quick_suggestions:
            return quick_suggestions[query_type]
        
        # Fallback to dynamic generation for unknown types
        try:
            suggestion_prompt = f"""
            Provide 3 example queries and best practices for {query_type} security analysis.
            Keep it concise and practical.
            """
            
            if conv_context:
                suggestion_prompt += f"\n\nConversation context:\n{conv_context[:300]}"

            messages = [
                SystemMessage(content="Provide practical security query suggestions."),
                HumanMessage(content=suggestion_prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Suggestion error: {str(e)}")
            return f"Query suggestions for {query_type}: Try specific field names, timeframes, and conditions for better results."

    def record_correction(self, original_nl: str, original_query: str, corrected_query: str, feedback: str = "", session_id: str = None) -> str:
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
            
            # This should be automatically recorded by the memory system
            logger.info("Correction recorded successfully")
            
            return f"Correction recorded successfully. This will help improve future query conversions. Thank you for the feedback!"
            
        except Exception as e:
            logger.error(f"Correction recording error: {e}")
            return "Unable to record correction at this time. Please try again later."

    def get_conversion_summary(self, session_id: str = None) -> str:
        """Generate conversion summary with session awareness."""
        try:
            today = date.today().strftime("%Y-%m-%d")
            
            # Try session-specific summary first if session_id provided
            if session_id:
                conversions = self.memory_search_tool(f"conversions {today}", session_id=session_id)
            else:
                conversions = self.memory_search_tool(f"conversions {today}")

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

    def create_tools(self) -> List[Tool]:
        """Create LangChain tools for the workflow."""
        logger.debug("Creating workflow tools...")

        # These tools are designed to be called by the workflow, not directly by the assistant
        tools = [
            Tool(
                name="memory_search",
                description="Search past queries, patterns, and conversation history",
                func=lambda x: self.memory_search_tool(x)
            ),
            Tool(
                name="convert_to_query", 
                description="Convert natural language to structured security queries using structured output",
                func=lambda x: self.convert_to_query_tool(x)
            ),
            Tool(
                name="analyze_patterns",
                description="Analyze patterns in query conversion history", 
                func=lambda x: self.analyze_patterns_tool(x, "week")
            ),
            Tool(
                name="get_suggestions",
                description="Provide query suggestions and examples",
                func=lambda x: self.get_suggestions_tool(x)
            ),
            Tool(
                name="record_correction",
                description="Record user corrections and feedback",
                func=lambda x: "Please provide correction details in the format: original_query -> corrected_query"
            ),
            Tool(
                name="get_summary",
                description="Generate conversion summary",
                func=lambda x: self.get_conversion_summary()
            )
        ]

        logger.debug(f"Created {len(tools)} workflow tools with structured output support")
        return tools

    # Helper methods for workflow integration
    def get_tool_by_name(self, tool_name: str) -> Optional[callable]:
        """Get a specific tool function by name for workflow use."""
        tool_map = {
            "memory_search": self.memory_search_tool,
            "convert_to_query": self.convert_to_query_tool,
            "analyze_patterns": self.analyze_patterns_tool,
            "get_suggestions": self.get_suggestions_tool,
            "record_correction": self.record_correction,
            "get_conversion_summary": self.get_conversion_summary
        }
        
        return tool_map.get(tool_name)

    def validate_tool_input(self, tool_name: str, input_data: Any) -> bool:
        """Validate input for a specific tool."""
        try:
            if tool_name == "memory_search":
                return isinstance(input_data, str) and len(input_data.strip()) > 0
            elif tool_name == "convert_to_query":
                return isinstance(input_data, str) and len(input_data.strip()) > 0
            elif tool_name == "analyze_patterns":
                return isinstance(input_data, str)
            elif tool_name == "get_suggestions":
                return isinstance(input_data, str)
            elif tool_name == "record_correction":
                return isinstance(input_data, (str, dict))
            elif tool_name == "get_conversion_summary":
                return True  # No specific validation needed
            
            return False
            
        except Exception as e:
            logger.error(f"Tool input validation error: {e}")
            return False

    # Additional helper methods for structured output
    def get_structured_conversion_with_confidence(
        self, 
        natural_language: str, 
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Get structured conversion result with confidence scoring."""
        try:
            # Use structured output to get detailed result
            result = self.structured_query_llm.invoke([
                SystemMessage(content="""Convert the security query and provide confidence scoring.
                
Return detailed conversion information including:
- The converted query
- Confidence score (0.0-1.0)  
- Query type classification
- Any relevant timestamps"""),
                HumanMessage(content=f"Convert: {natural_language}")
            ])
            
            if hasattr(result, 'result') and isinstance(result.result, QueryConversionSuccess):
                conversion = result.result
                return {
                    'success': True,
                    'query': conversion.query,
                    'confidence': conversion.confidence or 0.8,  # Default confidence
                    'query_type': conversion.query_type or 'general',
                    'from_timestamp': conversion.from_timestamp,
                    'to_timestamp': conversion.to_timestamp,
                    'meets_threshold': (conversion.confidence or 0.8) >= confidence_threshold
                }
            else:
                return {
                    'success': False,
                    'error': 'Conversion failed or out of scope',
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"Structured conversion with confidence error: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }

    def batch_convert_queries(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Convert multiple queries using structured output for efficiency."""
        results = []
        
        for i, query in enumerate(queries):
            try:
                logger.info(f"Processing batch query {i+1}/{len(queries)}")
                result = self.get_structured_conversion_with_confidence(query, **kwargs)
                results.append({
                    'index': i,
                    'original_query': query,
                    **result
                })
                
            except Exception as e:
                logger.error(f"Batch conversion error for query {i}: {e}")
                results.append({
                    'index': i,
                    'original_query': query,
                    'success': False,
                    'error': str(e),
                    'confidence': 0.0
                })
        
        return results

    def validate_structured_output_support(self) -> Dict[str, bool]:
        """Check which structured output features are supported by the current LLM."""
        capabilities = {
            'function_calling': False,
            'json_mode': False,
            'streaming': False,
            'pydantic_support': False
        }
        
        try:
            # Test function calling
            test_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccess,
                method="function_calling"
            )
            capabilities['function_calling'] = True
            capabilities['pydantic_support'] = True
        except:
            pass
            
        try:
            # Test JSON mode
            test_llm = self.finetuned_llm.with_structured_output(
                QueryConversionSuccessDict,
                method="json_mode"
            )
            capabilities['json_mode'] = True
        except:
            pass
            
        try:
            # Test streaming
            if hasattr(self.streaming_query_llm, 'stream'):
                capabilities['streaming'] = True
        except:
            pass
            
        logger.info(f"Structured output capabilities: {capabilities}")
        return capabilities

    def get_query_examples_structured(self, query_type: str = "general") -> List[Dict[str, str]]:
        """Get structured examples for different query types."""
        examples = {
            "network": [
                {
                    "natural_language": "Show me all connections to suspicious IP 192.168.1.100",
                    "structured_query": 'source_ip:"192.168.1.100" OR destination_ip:"192.168.1.100"',
                    "description": "Network connection analysis"
                },
                {
                    "natural_language": "Find DNS queries to malicious domains in the last hour",
                    "structured_query": 'query_type:"DNS" AND (domain:*malicious* OR threat_intel:true) AND timestamp:[now-1h TO now]',
                    "description": "DNS threat analysis with time constraint"
                }
            ],
            "process": [
                {
                    "natural_language": "Show processes running from temp directories",
                    "structured_query": 'process_path:(*temp* OR *tmp* OR *appdata\\local\\temp*)',
                    "description": "Process location analysis"
                },
                {
                    "natural_language": "Find unsigned executables running as system",
                    "structured_query": 'process_signed:false AND process_user:*system*',
                    "description": "Process integrity and privilege analysis"
                }
            ],
            "file": [
                {
                    "natural_language": "Show files created in system directories today",
                    "structured_query": 'file_path:(*system32* OR *windows*) AND file_created:[now/d TO now] AND event_type:file_create',
                    "description": "System file creation monitoring"
                }
            ]
        }
        
        return examples.get(query_type, examples["network"])

    def export_structured_config(self) -> Dict[str, Any]:
        """Export configuration for structured output setup."""
        capabilities = self.validate_structured_output_support()
        
        config = {
            'structured_output_enabled': any(capabilities.values()),
            'capabilities': capabilities,
            'schemas': {
                'success_schema': QueryConversionSuccess.schema(),
                'error_schema': QueryConversionError.schema(),
                'result_schema': QueryConversionResult.schema()
            },
            'field_context_enabled': bool(self.field_store),
            'memory_enabled': bool(self.memory_tool),
            'session_aware': hasattr(self.memory_system, 'search_session_memories')
        }
        
        logger.info("Structured output configuration exported")
        return config