from functools import lru_cache
from typing import Optional, List, Dict
from config import logger
import re

class FieldContextManager:
    """Manages field context and example-based learning."""
    
    def __init__(self, field_store, field_context_k: int = 5):
        self.field_store = field_store
        self.field_context_k = field_context_k
        self.example_context_k = 3
        self.example_relevance_threshold = 1.5

    @lru_cache(maxsize=100)
    def build_field_context(self, nl_query: str, k: Optional[int] = None) -> str:
        """Build field context from available field definitions."""
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

    @lru_cache(maxsize=100)
    def build_example_context(self, nl_query: str, k: Optional[int] = None) -> str:
        """Build example context using progressive query degradation."""
        try:
            k = k or self.example_context_k
            
            search_strategies = [
                self._extract_all_meaningful_words,
                self._extract_action_and_object_words,
                self._extract_domain_words,
                self._extract_core_concepts,
            ]
            
            all_examples = []
            seen_examples = set()
            
            for i, strategy in enumerate(search_strategies):
                search_query = strategy(nl_query)
                if not search_query:
                    continue
                    
                examples = self.field_store.search_examples(
                    search_query,
                    limit=k * 2,
                    relevance_threshold=self.example_relevance_threshold
                ) or []
                
                new_examples = []
                for ex in examples:
                    query_hash = hash(ex.get("query", "").strip().lower())
                    if query_hash not in seen_examples:
                        seen_examples.add(query_hash)
                        new_examples.append(ex)
                        all_examples.append(ex)
                
                if len(all_examples) >= k and i >= 1:
                    break
            
            if not all_examples:
                return ""
            
            ranked_examples = self._rank_examples_by_content_similarity(nl_query, all_examples)
            final_examples = ranked_examples[:k]
            
            return self._format_examples(final_examples, nl_query)
            
        except Exception as e:
            logger.error(f"Example context error: {e}")
            return ""

    def build_combined_context(self, nl_query: str) -> Dict[str, str]:
        """Build combined field and example context with optimization."""
        context = {
            'field_context': '',
            'example_context': '',
            'combined_context': ''
        }
        
        try:
            field_context = self.build_field_context(nl_query[:100])
            example_context = self.build_example_context(nl_query[:100])
            
            context['field_context'] = field_context
            context['example_context'] = example_context
            
            if field_context and example_context:
                limited_fields = field_context
                if len(field_context) > 300:
                    field_lines = field_context.split('\n')
                    limited_fields = '\n'.join(field_lines[:6])
                
                context['combined_context'] = f"{limited_fields}\n\n{example_context}"
            elif field_context:
                context['combined_context'] = field_context
            elif example_context:
                context['combined_context'] = example_context
            
        except Exception as e:
            logger.error(f"Combined context error: {e}")
        
        return context

    def _extract_all_meaningful_words(self, query: str) -> str:
        """Extract all potentially meaningful words, excluding only common stop words."""
        minimal_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query.lower())
        meaningful_words = [w for w in words if w not in minimal_stops and len(w) > 1]
        return " ".join(meaningful_words)

    def _extract_action_and_object_words(self, query: str) -> str:
        """Focus on action words and potential object words."""
        words = re.findall(r'\b\w+\b', query.lower())
        
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
        
        result_words = actions + objects + other[:5]
        return " ".join(result_words)

    def _extract_domain_words(self, query: str) -> str:
        """Extract words that are likely related to security/IT domain."""
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        domain_words = []
        for word in words:
            if (len(word) >= 3 and 
                word not in {'the', 'and', 'with', 'from', 'that', 'this', 'are', 'was', 'were', 'have', 'has'}):
                domain_words.append(word)
        
        return " ".join(domain_words)

    def _extract_core_concepts(self, query: str) -> str:
        """Extract the most abstract, core concepts."""
        query_lower = query.lower()
        
        time_indicators = []
        if any(term in query_lower for term in ['time', 'minute', 'hour', 'day', 'last', 'recent', 'past', 'ago']):
            time_indicators.extend(['time', 'temporal'])
        
        data_indicators = []
        if any(term in query_lower for term in ['select', 'find', 'show', 'get', 'search', 'query']):
            data_indicators.extend(['search', 'query'])
        
        if any(term in query_lower for term in ['log', 'event', 'record', 'data']):
            data_indicators.extend(['data', 'logs'])
        
        network_indicators = []
        if any(term in query_lower for term in ['network', 'connection', 'traffic', 'address', 'ip', 'port']):
            network_indicators.extend(['network'])
        
        concepts = time_indicators + data_indicators + network_indicators
        
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
            
            # Base vector similarity score
            distance = ex.get("distance", 1.0)
            base_score = max(0, 1.5 - distance)
            score += base_score * 2.0
            
            # Word overlap scoring
            ex_words = set(re.findall(r'\b\w+\b', combined_text))
            overlap = len(nl_words & ex_words)
            overlap_ratio = overlap / len(nl_words) if nl_words else 0
            score += overlap_ratio * 3.0
            
            # Partial word matching
            partial_matches = 0
            for nl_word in nl_words:
                if len(nl_word) > 3:
                    for ex_word in ex_words:
                        if nl_word in ex_word or ex_word in nl_word:
                            partial_matches += 0.5
            score += min(partial_matches, 2.0)
            
            # Length similarity bonus
            ex_length = len(f"{description} {query}".split())
            if ex_length > 0:
                length_ratio = min(nl_length, ex_length) / max(nl_length, ex_length)
                if length_ratio > 0.5:
                    score += 0.5
            
            # Query structure similarity
            nl_has_numbers = bool(re.search(r'\d', nl_query))
            ex_has_numbers = bool(re.search(r'\d', combined_text))
            if nl_has_numbers == ex_has_numbers:
                score += 0.3
            
            # Prefer examples with actual query syntax
            if ':' in query and len(query) > 10:
                score += 0.4
            
            # Penalty for overly long examples
            if len(combined_text) > 300:
                score -= 0.2
                
            scored_examples.append((score, ex))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored_examples]

    def _format_examples(self, examples: List[Dict], original_query: str) -> str:
        """Format examples for display."""
        if not examples:
            return ""
        
        example_lines = []
        original_words = set(re.findall(r'\b\w+\b', original_query.lower()))
        
        for i, ex in enumerate(examples, 1):
            description = ex.get("description", "").strip()
            query = ex.get("query", "").strip()
            distance = ex.get("distance", 1.0)
            
            combined_text = f"{description} {query}".lower()
            ex_words = set(re.findall(r'\b\w+\b', combined_text))
            matching_words = original_words & ex_words
            
            example_text = f"{i}. {description}"
            
            if len(query) <= 100:
                query_display = query
            else:
                query_display = query[:97] + "..."
            example_text += f"\n   Query: {query_display}"
            
            if matching_words and len(matching_words) <= 5:
                match_list = ", ".join(sorted(matching_words))
                example_text += f"\n   Matches: {match_list}"
            
            similarity = max(0, 1.5 - distance)
            if similarity > 0.1:
                example_text += f"\n   Relevance: {similarity:.2f}"
            
            example_lines.append(example_text)
        
        header = f"Found {len(examples)} relevant examples:"
        return f"{header}\n\n" + "\n\n".join(example_lines)