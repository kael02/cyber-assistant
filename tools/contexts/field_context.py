from functools import lru_cache
from typing import Optional, List, Dict, Set, Tuple
from config import logger
import re
import string


class FieldContextManager:
    """
    Flexible field + example context manager that adapts to any domain.
    - Generic tokenization that doesn't assume specific data types
    - Configurable entity extraction patterns
    - Adaptive search strategies based on query characteristics
    - Domain-agnostic ranking and scoring
    """

    def __init__(self, field_store, field_context_k: int = 5, **config):
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.example_context_k = config.get('example_context_k', 3)
        self.example_relevance_threshold = config.get('relevance_threshold', 1.5)
        self.max_search_calls = config.get('max_search_calls', 3)
        
        # Configurable patterns - can be extended or overridden
        self.entity_patterns = self._init_entity_patterns(config.get('custom_patterns', {}))
        self.stopwords = self._init_stopwords(config.get('custom_stopwords', set()))

    def _init_entity_patterns(self, custom_patterns: Dict[str, str]) -> Dict[str, re.Pattern]:
        """Initialize entity extraction patterns. Easily extensible."""
        default_patterns = {
            'field_reference': r'[a-zA-Z_][a-zA-Z0-9_]*:',  # field: syntax
            'quoted_string': r'"[^"]*"|\'[^\']*\'',  # quoted values
            'numeric_value': r'\b\d+(?:\.\d+)?\b',  # numbers
            'identifier': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  # identifiers
            'special_chars': r'[=<>!]+',  # operators
            'bracketed': r'\[[^\]]*\]|\([^)]*\)',  # bracketed content
        }
        
        # Merge with custom patterns
        all_patterns = {**default_patterns, **custom_patterns}
        return {name: re.compile(pattern, re.IGNORECASE) for name, pattern in all_patterns.items()}

    def _init_stopwords(self, custom_stopwords: Set[str]) -> Set[str]:
        """Initialize stopwords. Easily customizable per domain."""
        default_stops = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'that', 'this', 'is', 'are', 'was', 'were',
            'show', 'find', 'get', 'all', 'data', 'where', 'when', 'how', 'what'
        }
        return default_stops.union(custom_stopwords)

    # ---------- Public API ----------

    @lru_cache(maxsize=100)
    def build_field_context(self, nl_query: str, k: Optional[int] = None) -> str:
        """Build field context using semantic search."""
        try:
            k = k or self.field_context_k
            hits = self.field_store.search_fields(nl_query, limit=k) or []
            if not hits:
                return ""

            return self._format_field_context(hits[:k])
        except Exception as e:
            logger.error(f"Field context error: {e}")
            return ""

    @lru_cache(maxsize=100)
    def build_example_context(self, nl_query: str, k: Optional[int] = None) -> str:
        """
        Build example context using adaptive search strategy:
        1. Direct query (user's exact intent)
        2. Enhanced query (with extracted entities)
        3. Simplified query (key terms only)
        """
        try:
            k = k or self.example_context_k
            all_examples: List[Dict] = []
            seen: Set[str] = set()
            search_calls = 0

            def add_unique_examples(results: List[Dict]):
                if not results:
                    return
                for ex in results:
                    query_key = self._get_example_key(ex)
                    if query_key not in seen:
                        seen.add(query_key)
                        all_examples.append(ex)

            # Strategy 1: Direct search with original query
            if search_calls < self.max_search_calls:
                try:
                    results = self.field_store.search_examples(
                        nl_query,
                        limit=max(k * 2, 10),
                        relevance_threshold=self.example_relevance_threshold
                    ) or []
                    add_unique_examples(results)
                    search_calls += 1
                except Exception as e:
                    logger.warning(f"Direct search failed: {e}")

            # Strategy 2: Enhanced search (if we need more examples)
            if len(all_examples) < k and search_calls < self.max_search_calls:
                enhanced_query = self._create_enhanced_query(nl_query)
                if enhanced_query != nl_query:  # Only search if different
                    try:
                        results = self.field_store.search_examples(
                            enhanced_query,
                            limit=max(k * 2, 10),
                            relevance_threshold=self.example_relevance_threshold
                        ) or []
                        add_unique_examples(results)
                        search_calls += 1
                    except Exception as e:
                        logger.warning(f"Enhanced search failed: {e}")

            # Strategy 3: Simplified search (if still need more)
            if len(all_examples) < k and search_calls < self.max_search_calls:
                simplified_query = self._create_simplified_query(nl_query)
                if simplified_query and simplified_query not in {nl_query, enhanced_query}:
                    try:
                        results = self.field_store.search_examples(
                            simplified_query,
                            limit=max(k * 2, 10),
                            relevance_threshold=self.example_relevance_threshold
                        ) or []
                        add_unique_examples(results)
                        search_calls += 1
                    except Exception as e:
                        logger.warning(f"Simplified search failed: {e}")

            if not all_examples:
                return ""

            # Rank and return top examples
            ranked = self._rank_examples(nl_query, all_examples)
            return self._format_examples(ranked[:k], nl_query)

        except Exception as e:
            logger.error(f"Example context error: {e}")
            return ""

    def build_combined_context(self, nl_query: str) -> Dict[str, str]:
        """Build combined context efficiently."""
        context = {"field_context": "", "example_context": "", "combined_context": ""}

        try:
            # Truncate very long queries for efficiency
            query_for_context = nl_query[:200] if len(nl_query) > 200 else nl_query
            
            field_context = self.build_field_context(query_for_context)
            example_context = self.build_example_context(query_for_context)

            context["field_context"] = field_context
            context["example_context"] = example_context

            # Combine contexts intelligently
            if field_context and example_context:
                # Limit field context if it's too long
                limited_field = self._limit_field_context(field_context, max_lines=6)
                context["combined_context"] = f"{limited_field}\n\n{example_context}"
            else:
                context["combined_context"] = field_context or example_context

        except Exception as e:
            logger.error(f"Combined context error: {e}")

        return context

    # ---------- Query Processing ----------

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using configurable patterns."""
        if not text:
            return {}

        entities = {}
        text_lower = text.lower()

        for pattern_name, pattern in self.entity_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                # Clean up matches
                cleaned_matches = []
                for match in matches:
                    cleaned = match.strip(string.punctuation + string.whitespace)
                    if cleaned and len(cleaned) > 1:
                        cleaned_matches.append(cleaned)
                
                if cleaned_matches:
                    entities[pattern_name] = list(dict.fromkeys(cleaned_matches))  # Remove duplicates

        return entities

    def _create_enhanced_query(self, nl_query: str) -> str:
        """Create an enhanced query by adding important extracted entities."""
        entities = self._extract_entities(nl_query)
        
        # Prioritize certain types of entities
        priority_types = ['field_reference', 'quoted_string', 'identifier']
        enhancement_parts = []
        
        for entity_type in priority_types:
            if entity_type in entities:
                enhancement_parts.extend(entities[entity_type][:3])  # Limit to avoid bloat
        
        # Add other entity types if we don't have enough
        if len(enhancement_parts) < 5:
            for entity_type, values in entities.items():
                if entity_type not in priority_types:
                    enhancement_parts.extend(values[:2])
                if len(enhancement_parts) >= 8:
                    break
        
        if enhancement_parts:
            return f"{nl_query} {' '.join(enhancement_parts[:8])}"
        return nl_query

    def _create_simplified_query(self, nl_query: str) -> str:
        """Create a simplified query with just key terms."""
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', nl_query.lower())
        
        # Remove stopwords and short tokens
        meaningful_tokens = [
            token for token in tokens 
            if token not in self.stopwords and len(token) > 2
        ]
        
        # Prioritize longer tokens and potential field names
        scored_tokens = []
        for token in meaningful_tokens:
            score = len(token)  # Longer tokens get higher scores
            if '_' in token or token.endswith('id'):  # Potential field names
                score += 5
            scored_tokens.append((score, token))
        
        # Sort by score and take top tokens
        scored_tokens.sort(reverse=True)
        top_tokens = [token for _, token in scored_tokens[:8]]
        
        return ' '.join(top_tokens) if top_tokens else ""

    # ---------- Ranking ----------

    def _rank_examples(self, nl_query: str, examples: List[Dict]) -> List[Dict]:
        """Generic ranking based on semantic similarity and term overlap."""
        if not examples:
            return []

        query_tokens = self._tokenize_for_matching(nl_query)
        query_entities = self._extract_entities(nl_query)

        def score_example(ex: Dict) -> float:
            score = 0.0
            
            # Base score from vector similarity
            distance = ex.get("distance", 1.0)
            score += max(0, 2.0 - distance) * 2.0
            
            # Text content analysis
            desc = ex.get("description", "")
            query = ex.get("query", "")
            combined_text = f"{desc} {query}".lower()
            
            ex_tokens = self._tokenize_for_matching(combined_text)
            ex_entities = self._extract_entities(combined_text)
            
            # Token overlap score
            if query_tokens and ex_tokens:
                overlap = len(query_tokens.intersection(ex_tokens))
                score += (overlap / len(query_tokens)) * 3.0
            
            # Entity alignment score
            for entity_type, query_vals in query_entities.items():
                ex_vals = ex_entities.get(entity_type, [])
                if query_vals and ex_vals:
                    matches = len(set(query_vals).intersection(set(ex_vals)))
                    score += matches * 2.0
            
            # Quality indicators
            if len(query.strip()) > 10:  # Prefer examples with substantial queries
                score += 0.5
            
            category = ex.get("category", "")
            complexity = ex.get("complexity", "")
            if category and complexity:  # Well-categorized examples
                score += 0.3
            
            return score

        scored_examples = [(score_example(ex), ex) for ex in examples]
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        return [ex for _, ex in scored_examples]

    def _tokenize_for_matching(self, text: str) -> Set[str]:
        """Simple tokenization for matching purposes."""
        if not text:
            return set()
        
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {token for token in tokens if token not in self.stopwords and len(token) > 1}

    # ---------- Formatting ----------

    def _format_field_context(self, hits: List[Dict]) -> str:
        """Format field context in a clean way."""
        if not hits:
            return ""

        lines = []
        for hit in hits:
            alias = (hit.get("alias") or "").strip()
            long_name = (hit.get("long_name") or "")
            data_type = (hit.get("data_type") or "").strip()
            
            # Truncate long names sensibly
            if long_name and len(long_name) > 100:
                long_name = long_name[:97] + "..."
            
            parts = [alias, data_type, long_name]
            parts = [p for p in parts if p]  # Remove empty parts
            
            if parts:
                lines.append(f"- {' | '.join(parts)}")
        
        return f"Available fields:\n" + "\n".join(lines) if lines else ""

    def _format_examples(self, examples: List[Dict], original_query: str) -> str:
        """Format examples in a readable way."""
        if not examples:
            return ""

        lines = []
        original_tokens = self._tokenize_for_matching(original_query)

        for i, ex in enumerate(examples, 1):
            desc = (ex.get("description") or "").strip()
            query = (ex.get("query") or "").strip()
            category = (ex.get("category") or "").strip()
            complexity = (ex.get("complexity") or "").strip()
            distance = ex.get("distance", 1.0)
            
            # Main description
            text = f"{i}. {desc}" if desc else f"{i}. [No description]"
            
            # Add metadata if available
            metadata_parts = []
            if category:
                metadata_parts.append(f"Category: {category}")
            if complexity:
                metadata_parts.append(f"Complexity: {complexity}")
            
            if metadata_parts:
                text += f"\n   ({', '.join(metadata_parts)})"
            
            # Show query (truncate if very long)
            if query:
                if len(query) <= 150:
                    text += f"\n   Query: {query}"
                else:
                    text += f"\n   Query: {query[:147]}..."
            
            # Show matching terms
            if query and original_tokens:
                ex_tokens = self._tokenize_for_matching(f"{desc} {query}")
                matches = original_tokens.intersection(ex_tokens)
                if matches:
                    # Show most relevant matches
                    sorted_matches = sorted(matches, key=len, reverse=True)
                    text += f"\n   Key matches: {', '.join(sorted_matches[:6])}"
            
            # Show relevance if meaningful
            relevance = max(0, 2.0 - distance)
            if relevance > 0.2:
                text += f"\n   Relevance: {relevance:.2f}"
            
            lines.append(text)

        header = f"Found {len(examples)} relevant example{'s' if len(examples) != 1 else ''}:"
        return f"{header}\n\n" + "\n\n".join(lines)

    def _limit_field_context(self, field_context: str, max_lines: int = 6) -> str:
        """Limit field context to prevent overwhelming the output."""
        if not field_context:
            return field_context
            
        lines = field_context.split('\n')
        if len(lines) <= max_lines + 1:  # +1 for header
            return field_context
            
        limited_lines = lines[:max_lines + 1]
        remaining = len(lines) - len(limited_lines)
        if remaining > 0:
            limited_lines.append(f"   ... and {remaining} more fields")
            
        return '\n'.join(limited_lines)

    def _get_example_key(self, example: Dict) -> str:
        """Generate a unique key for deduplication."""
        query = example.get("query", "")
        desc = example.get("description", "")
        return f"{query}|{desc}".lower().strip()