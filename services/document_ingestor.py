import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from config import get_settings, logger


class DocumentIngestor:
    """
    Stores field definitions and query examples in Postgres+pgvector collections via LangChain.
    Uses Qwen3 embeddings via Hugging Face - latest embedding model from Qwen family.
    """

    FIELD_COLLECTION = "field_definitions"
    EXAMPLE_COLLECTION = "query_examples"

    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None, huggingface_api_token: Optional[str] = None):
        logger.info("🔧 Initializing Vector Field Store with Qwen3 embeddings...")

        self.settings = get_settings()
        self.connection_url = (
            f"postgresql+psycopg://{self.settings.DATABASE_USER}:"
            f"{self.settings.DATABASE_PASSWORD}@{self.settings.DATABASE_HOST}:"
            f"{self.settings.DATABASE_PORT}/{self.settings.DATABASE_NAME}"
        )

        try:
            # Use latest Qwen3 embeddings - API-based, no downloads required
            if embeddings is None:
                logger.info("🧠 Initializing Qwen3 embeddings...")
                # Get API token from parameter, environment, or settings
                api_token = (
                    huggingface_api_token or 
                    os.getenv("HUGGINGFACE_API_TOKEN") or 
                    getattr(self.settings, "HUGGINGFACE_API_TOKEN", None)
                )
                
                if not api_token:
                    raise ValueError(
                        "Hugging Face API token required. Set HUGGINGFACE_API_TOKEN environment variable "
                        "or pass huggingface_api_token parameter. Get free token at: https://huggingface.co/settings/tokens"
                    )
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="Qwen/Qwen3-Embedding-0.6B",  # Latest Qwen3 embedding model (768 dimensions)
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': True,
                        'token': api_token
                    },
                    encode_kwargs={'normalize_embeddings': True}  # Normalized embeddings for better similarity search
                )
                logger.info("✅ Qwen3 embeddings initialized successfully")
            else:
                self.embeddings = embeddings
            
            # Field definitions vector store
            self.field_vs = PGVector(
                embeddings=self.embeddings,
                collection_name=self.FIELD_COLLECTION,
                connection=self.connection_url,
                use_jsonb=True,
                create_extension=True,
            )

            # Query examples vector store  
            self.example_vs = PGVector(
                embeddings=self.embeddings,
                collection_name=self.EXAMPLE_COLLECTION,
                connection=self.connection_url,
                use_jsonb=True,
                create_extension=True,
            )

            self.engine = create_engine(
                self.connection_url,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Test connections
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vector Field Store: {str(e)}")
            raise

        logger.info("✅ Vector Field Store initialized with Qwen3 embeddings")

    # === SEARCH WITH OPTIMIZED TASK TYPE ===
    
    def _create_search_embeddings(self):
        """Create embeddings optimized for search queries."""
        api_token = (
            os.getenv("HUGGINGFACE_API_TOKEN") or 
            getattr(self.settings, "HUGGINGFACE_API_TOKEN", None)
        )
        return HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",  # Latest Qwen3 embedding model
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True,
                'token': api_token
            },
            encode_kwargs={'normalize_embeddings': True}
        )

    # === CLEAR COLLECTIONS (NEW METHOD) ===
    
    def clear_all_collections(self) -> Dict[str, Any]:
        """Clear all existing vectors to resolve dimension mismatches."""
        logger.info("🗑️ Clearing all collections to resolve dimension mismatches...")
        try:
            # Delete all embeddings from both collections
            sql = text("""
                DELETE FROM langchain_pg_embedding e
                USING langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name IN (:field_col, :example_col)
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(sql, {
                    "field_col": self.FIELD_COLLECTION, 
                    "example_col": self.EXAMPLE_COLLECTION
                })
                deleted = result.rowcount or 0
                
            logger.info(f"✅ Cleared {deleted} records from all collections")
            return {"success": True, "deleted_count": deleted}
            
        except Exception as e:
            error_msg = f"Error clearing collections: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    # === FIELD DEFINITIONS ===
    
    @staticmethod
    def _expected_field_columns() -> List[str]:
        return [
            "Old Alias",
            "Alias", 
            "Long Name",
            "Definition",
            "Data Type",
            "Example Value",
            "Remark",
        ]

    @staticmethod
    def _field_doc_from_row(row: Dict[str, str], source_filename: str) -> Document:
        """Build a LangChain Document optimized for semantic search of field definitions."""
        alias = (row.get("Alias") or "").strip()
        long_name = (row.get("Long Name") or "").strip()
        definition = (row.get("Definition") or "").strip()
        data_type = (row.get("Data Type") or "").strip()
        example = (row.get("Example Value") or "").strip()
        remark = (row.get("Remark") or "").strip()
        old_alias = (row.get("Old Alias") or "").strip()

        # Create natural language content for semantic search
        content_parts = []
        
        if alias:
            content_parts.append(f"Field name {alias}")
            
        if long_name and long_name != alias:
            content_parts.append(f"also called {long_name}")
            
        if definition:
            content_parts.append(f"is {definition}")
        
        # Add data type context
        if data_type:
            type_descriptions = {
                'ip': 'an IP address field',
                'uuid': 'a unique identifier field', 
                'int32': 'an integer number field',
                'string': 'a text field',
                'timestamp': 'a time/date field',
                'port': 'a network port field',
                'url': 'a web address field',
                'email': 'an email address field'
            }
            type_desc = type_descriptions.get(data_type.lower(), f'a {data_type} field')
            content_parts.append(f"This is {type_desc}")
        
        # Add semantic keywords based on field purpose
        field_text = f"{alias} {long_name} {definition}".lower()
        semantic_keywords = []
        
        if any(term in field_text for term in ['ip', 'address']):
            if 'src' in field_text or 'source' in field_text:
                semantic_keywords.append("source IP address")
            elif 'dst' in field_text or 'dest' in field_text or 'target' in field_text:
                semantic_keywords.append("destination IP address")
            else:
                semantic_keywords.append("IP address")
        
        if 'port' in field_text:
            if 'src' in field_text or 'source' in field_text:
                semantic_keywords.append("source port")
            elif 'dst' in field_text or 'dest' in field_text:
                semantic_keywords.append("destination port")
            else:
                semantic_keywords.append("network port")
        
        if any(term in field_text for term in ['time', 'date', 'year', 'month', 'day', 'receive']):
            semantic_keywords.append("timestamp")
        
        if any(term in field_text for term in ['user', 'account', 'login']):
            semantic_keywords.append("user identity")
            
        if any(term in field_text for term in ['event', 'action', 'activity']):
            semantic_keywords.append("event information")
            
        if any(term in field_text for term in ['host', 'device', 'machine']):
            semantic_keywords.append("device information")
        
        if semantic_keywords:
            content_parts.append(f"Used for {', '.join(semantic_keywords)}")
        
        if example and len(example) < 50:
            content_parts.append(f"Example value: {example}")
            
        page_content = ". ".join(content_parts).replace(".. ", ". ")
        if not page_content.endswith('.'):
            page_content += "."

        metadata = {
            "type": "field_definition",
            "alias": alias,
            "old_alias": old_alias or None,
            "long_name": long_name or None,
            "definition": definition or None,
            "data_type": data_type or None,
            "example_value": example or None,
            "remark": remark or None,
            "source_file": source_filename,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        
        return Document(page_content=page_content, metadata=metadata)

    def ingest_field_definitions(self, file_path: str) -> Dict[str, Any]:
        """Ingest field definitions from CSV file."""
        logger.info(f"📖 Starting field definitions ingestion: {file_path}")

        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": 0}

        processed = 0
        skipped = 0
        errors: List[str] = []
        source_filename = Path(file_path).name

        try:
            # Remove existing records from this source
            delete_result = self.delete_fields_by_source_file(source_filename)
            if delete_result.get("deleted_count", 0) > 0:
                logger.info(f"Removed {delete_result['deleted_count']} existing field records")

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Validate required columns
                missing = [c for c in self._expected_field_columns() if c not in (reader.fieldnames or [])]
                if missing:
                    error_msg = f"Missing columns: {missing}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "processed": 0}

                docs: List[Document] = []
                ids: List[str] = []
                batch_size = 50  # Smaller batch size for Hugging Face models

                for row_num, row in enumerate(reader, start=2):
                    try:
                        # Skip empty rows
                        if not any(row.values()):
                            skipped += 1
                            continue
                            
                        alias = (row.get("Alias") or "").strip()
                        if not alias:
                            logger.warning(f"Row {row_num}: Missing alias, skipping")
                            skipped += 1
                            continue

                        doc = self._field_doc_from_row(row, source_filename)
                        docs.append(doc)
                        ids.append(self._field_id_for(alias, source_filename))

                        # Process in batches
                        if len(docs) >= batch_size:
                            try:
                                logger.info(f"Processing batch of {len(docs)} field documents...")
                                self.field_vs.add_documents(docs, ids=ids)
                                processed += len(docs)
                                docs.clear()
                                ids.clear()
                                logger.info(f"Processed {processed} field records...")
                            except Exception as batch_error:
                                logger.error(f"Batch error: {str(batch_error)}")
                                errors.append(f"Batch error at row {row_num}: {str(batch_error)}")
                                docs.clear()
                                ids.clear()

                    except Exception as e:
                        msg = f"Row {row_num}: {str(e)}"
                        logger.error(msg)
                        errors.append(msg)
                        skipped += 1

                # Process final batch
                if docs:
                    try:
                        logger.info(f"Processing final batch of {len(docs)} field documents...")
                        self.field_vs.add_documents(docs, ids=ids)
                        processed += len(docs)
                    except Exception as final_error:
                        logger.error(f"Final batch error: {str(final_error)}")
                        errors.append(f"Final batch error: {str(final_error)}")

            logger.info(f"✅ Field ingestion completed - Processed: {processed}, Skipped: {skipped}")
            return {
                "success": True,
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "source_file": source_filename,
            }

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": processed}

    # === QUERY EXAMPLES ===
    
    @staticmethod
    def _expected_example_columns() -> List[str]:
        return ["description", "query"]

    @staticmethod 
    def _example_doc_from_row(row: Dict[str, str], source_filename: str) -> Document:
        """Build a LangChain Document optimized for semantic search of query examples."""
        # Get the core required fields
        description = (row.get("description") or "").strip()
        query = (row.get("query") or "").strip()
        
        # Get optional fields - these may not exist in all CSV files
        category = (row.get("category") or "").strip()
        complexity = (row.get("complexity") or "").strip()  
        fields_used = (row.get("fields_used") or "").strip()

        # Create natural language content for semantic search
        content_parts = []
        
        if description:
            content_parts.append(f"Query description: {description}")
            
        if category:
            content_parts.append(f"Category: {category}")
            
        if complexity:
            content_parts.append(f"Complexity level: {complexity}")
            
        if fields_used:
            content_parts.append(f"Uses these fields: {fields_used}")
        
        # Extract semantic keywords from description and query for better search
        combined_text = f"{description} {query}".lower()
        semantic_keywords = []
        
        # Network-related patterns
        if any(term in combined_text for term in ['ip', 'network', 'traffic', 'connection', 'srcip', 'trgip']):
            semantic_keywords.append("network analysis")
            
        # Security patterns  
        if any(term in combined_text for term in ['malware', 'threat', 'attack', 'suspicious', 'block', 'malicious']):
            semantic_keywords.append("security analysis")
            
        # Event patterns
        if any(term in combined_text for term in ['event', 'log', 'audit', 'activity', 'evt:']):
            semantic_keywords.append("event analysis")
            
        # User patterns
        if any(term in combined_text for term in ['user', 'account', 'login', 'authentication', 'srcuname', 'trguname']):
            semantic_keywords.append("user analysis")
            
        # Process patterns
        if any(term in combined_text for term in ['process', 'executable', 'program', 'application', 'pname:']):
            semantic_keywords.append("process analysis")
            
        # Time-based patterns
        if any(term in combined_text for term in ['time', 'date', 'recent', 'last', 'between']):
            semantic_keywords.append("temporal analysis")
            
        # Tenant-based patterns
        if 'tenant' in combined_text or 'tenantname:' in combined_text:
            semantic_keywords.append("tenant filtering")
            
        # Domain patterns
        if any(term in combined_text for term in ['domain', 'trgdomain:', 'dns']):
            semantic_keywords.append("domain analysis")
            
        if semantic_keywords:
            content_parts.append(f"Analysis type: {', '.join(semantic_keywords)}")
        
        # Extract field names from the query for better matching
        if query:
            import re
            field_patterns = re.findall(r'(\w+):', query)
            if field_patterns:
                unique_fields = list(set(field_patterns))
                content_parts.append(f"Query fields: {', '.join(unique_fields)}")
        
        # Add query terms for semantic matching
        if query:
            import re
            terms = re.findall(r'\b[a-zA-Z0-9_\.]+\b', query)
            meaningful_terms = [t for t in terms if len(t) > 2 and t.lower() not in 
                            {'and', 'or', 'not', 'with', 'from', 'where', 'select', 'evt', 'com'}][:10]
            if meaningful_terms:
                content_parts.append(f"Key terms: {', '.join(meaningful_terms)}")
        
        page_content = ". ".join(content_parts)
        if not page_content.endswith('.'):
            page_content += "."

        metadata = {
            "type": "query_example",
            "description": description,
            "query": query,
            "category": category or None,
            "complexity": complexity or None, 
            "fields_used": fields_used or None,
            "source_file": source_filename,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        
        return Document(page_content=page_content, metadata=metadata)

    def ingest_query_examples(self, file_path: str) -> Dict[str, Any]:
        """Ingest query examples from CSV file."""
        logger.info(f"📖 Starting query examples ingestion: {file_path}")

        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": 0}

        processed = 0
        skipped = 0
        errors: List[str] = []
        source_filename = Path(file_path).name

        try:
            # Remove existing records from this source
            delete_result = self.delete_examples_by_source_file(source_filename)
            if delete_result.get("deleted_count", 0) > 0:
                logger.info(f"Removed {delete_result['deleted_count']} existing example records")

            with open(file_path, "r", encoding="utf-8") as f:
                # Use csv.Sniffer to detect dialect
                sample = f.read(1024)
                f.seek(0)
                
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except:
                    dialect = csv.excel
                
                reader = csv.DictReader(f, dialect=dialect)
                
                logger.info(f"Detected CSV columns: {reader.fieldnames}")
                
                # Validate required columns
                required = ["description", "query"]
                missing = [c for c in required if c not in (reader.fieldnames or [])]
                if missing:
                    error_msg = f"Missing required columns: {missing}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "processed": 0}

                docs: List[Document] = []
                ids: List[str] = []
                batch_size = 50  # Smaller batch size for Hugging Face models

                for row_num, row in enumerate(reader, start=2):
                    try:
                        if row_num <= 4:
                            logger.info(f"Row {row_num} data: {dict(row)}")
                        
                        # Skip empty rows
                        if not any(v.strip() for v in row.values() if v):
                            skipped += 1
                            continue
                            
                        description = (row.get("description") or "").strip()
                        query = (row.get("query") or "").strip()
                        
                        if not description:
                            logger.warning(f"Row {row_num}: Missing description, skipping")
                            skipped += 1
                            continue
                            
                        if not query:
                            logger.warning(f"Row {row_num}: Missing query, skipping")
                            skipped += 1
                            continue

                        doc = self._example_doc_from_row(row, source_filename)
                        docs.append(doc)
                        ids.append(self._example_id_for(description, source_filename, row_num))

                        # Process in batches
                        if len(docs) >= batch_size:
                            try:
                                logger.info(f"Processing batch of {len(docs)} example documents...")
                                self.example_vs.add_documents(docs, ids=ids)
                                processed += len(docs)
                                docs.clear()
                                ids.clear()
                                logger.info(f"Processed {processed} example records...")
                            except Exception as batch_error:
                                logger.error(f"Batch error: {str(batch_error)}")
                                errors.append(f"Batch error at row {row_num}: {str(batch_error)}")
                                docs.clear()
                                ids.clear()

                    except Exception as e:
                        msg = f"Row {row_num}: {str(e)}"
                        logger.error(msg)
                        errors.append(msg)
                        skipped += 1

                # Process final batch
                if docs:
                    try:
                        logger.info(f"Processing final batch of {len(docs)} example documents...")
                        self.example_vs.add_documents(docs, ids=ids)
                        processed += len(docs)
                    except Exception as final_error:
                        logger.error(f"Final batch error: {str(final_error)}")
                        errors.append(f"Final batch error: {str(final_error)}")

            logger.info(f"✅ Example ingestion completed - Processed: {processed}, Skipped: {skipped}")
            return {
                "success": True,
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "source_file": source_filename,
            }

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": processed}

    # === SEARCH METHODS ===

    def search_fields(self, query: str, limit: int = 5, relevance_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Vector similarity search for field definitions with relevance filtering."""
        logger.info(f"🔍 Searching fields for: '{query}'")
        try:
            hits_with_scores = self.field_vs.similarity_search_with_score(query, k=limit)
            
            results = []
            for doc, distance in hits_with_scores:
                if distance > relevance_threshold:
                    continue
                    
                md = doc.metadata or {}
                results.append({
                    "alias": md.get("alias"),
                    "old_alias": md.get("old_alias"),
                    "long_name": md.get("long_name"),
                    "definition": md.get("definition"),
                    "data_type": md.get("data_type"),
                    "example_value": md.get("example_value"),
                    "remark": md.get("remark"),
                    "source_file": md.get("source_file"),
                    "ingested_at": md.get("ingested_at"),
                    "distance": distance,
                })
                
            logger.info(f"✅ Found {len(results)} matching fields")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching fields: {str(e)}")
            return []

    def search_examples(self, query: str, limit: int = 5, relevance_threshold: float = 1.0, 
                       category: Optional[str] = None, complexity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Vector similarity search for query examples with optional filtering."""
        logger.info(f"🔍 Searching examples for: '{query}'")
        try:
            # Build filter if needed
            search_filter = {}
            if category:
                search_filter["category"] = {"$eq": category}
            if complexity:
                search_filter["complexity"] = {"$eq": complexity}
                
            hits_with_scores = self.example_vs.similarity_search_with_score(
                query, k=limit, filter=search_filter if search_filter else None
            )
            
            results = []
            for doc, distance in hits_with_scores:
                if distance > relevance_threshold:
                    continue
                    
                md = doc.metadata or {}
                results.append({
                    "description": md.get("description"),
                    "query": md.get("query"),
                    "category": md.get("category"),
                    "complexity": md.get("complexity"),
                    "fields_used": md.get("fields_used"),
                    "source_file": md.get("source_file"),
                    "ingested_at": md.get("ingested_at"),
                    "distance": distance,
                })
                
            logger.info(f"✅ Found {len(results)} matching examples")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching examples: {str(e)}")
            return []

    def search_combined(self, query: str, field_limit: int = 3, example_limit: int = 3, 
                       relevance_threshold: float = 1.0) -> Dict[str, Any]:
        """Search both fields and examples in one call for comprehensive results."""
        logger.info(f"🔍 Combined search for: '{query}'")
        
        fields = self.search_fields(query, field_limit, relevance_threshold)
        examples = self.search_examples(query, example_limit, relevance_threshold)
        
        return {
            "fields": fields,
            "examples": examples,
            "total_results": len(fields) + len(examples)
        }

    # === ID GENERATION ===
    
    @staticmethod
    def _field_id_for(alias: str, source_filename: str) -> str:
        """Deterministic ID for field definitions to prevent duplicates."""
        return f"field::{source_filename}::{alias.lower()}"
        
    @staticmethod  
    def _example_id_for(description: str, source_filename: str, row_num: int) -> str:
        """Deterministic ID for query examples to prevent duplicates."""
        desc_hash = hash(description.lower())
        return f"example::{source_filename}::{row_num}::{desc_hash}"

    # === DELETE METHODS ===
    
    def delete_fields_by_source_file(self, source_file: str) -> Dict[str, Any]:
        """Delete all field definition records from a specific source file."""
        logger.info(f"🗑️ Deleting fields from source file: {source_file}")
        try:
            sql = text("""
                DELETE FROM langchain_pg_embedding e
                USING langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name = :col
                  AND e.cmetadata->>'source_file' = :src
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(sql, {"col": self.FIELD_COLLECTION, "src": source_file})
                deleted = result.rowcount or 0
                
            logger.info(f"✅ Deleted {deleted} field records from {source_file}")
            return {"success": True, "deleted_count": deleted}
            
        except Exception as e:
            error_msg = f"Error deleting field records: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


    def delete_examples_by_source_file(self, source_file: str) -> Dict[str, Any]:
        """Delete all query example records from a specific source file."""
        logger.info(f"🗑️ Deleting examples from source file: {source_file}")
        try:
            sql = text("""
                DELETE FROM langchain_pg_embedding e
                USING langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name = :col
                  AND e.cmetadata->>'source_file' = :src
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(sql, {"col": self.EXAMPLE_COLLECTION, "src": source_file})
                deleted = result.rowcount or 0
                
            logger.info(f"✅ Deleted {deleted} example records from {source_file}")
            return {"success": True, "deleted_count": deleted}
            
        except Exception as e:
            error_msg = f"Error deleting example records: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}