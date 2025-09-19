import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from config import get_settings, logger


class DocumentIngestor:
    """
    Stores field definitions in a Postgres+pgvector collection via LangChain.
    Provides ingest/search/get/delete operations.
    """

    COLLECTION = "field_definitions"

    def __init__(self, embeddings: Optional[OpenAIEmbeddings] = None):
        logger.info("🔧 Initializing Vector Field Store...")

        self.settings = get_settings()
        self.connection_url = (
            f"postgresql+psycopg://{self.settings.DATABASE_USER}:"
            f"{self.settings.DATABASE_PASSWORD}@{self.settings.DATABASE_HOST}:"
            f"{self.settings.DATABASE_PORT}/{self.settings.DATABASE_NAME}"
        )

        try:
            self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-large")
            
            self.vs = PGVector(
                embeddings=self.embeddings,
                collection_name=self.COLLECTION,
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

        logger.info("✅ Vector Field Store initialized")

    @staticmethod
    def _expected_columns() -> List[str]:
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
    def _doc_from_row(row: Dict[str, str], source_filename: str) -> Document:
        """Build a LangChain Document optimized for semantic search."""
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

    @staticmethod
    def _id_for(alias: str, source_filename: str) -> str:
        """Deterministic ID to prevent duplicates."""
        return f"{source_filename}::{alias.lower()}"

    def ingest_field_definitions(self, file_path: str) -> Dict[str, Any]:
        """Ingest field definitions from CSV file."""
        logger.info(f"📖 Starting ingestion of file: {file_path}")

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
            delete_result = self.delete_by_source_file(source_filename)
            if delete_result.get("deleted_count", 0) > 0:
                logger.info(f"Removed {delete_result['deleted_count']} existing records")

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Validate required columns
                missing = [c for c in self._expected_columns() if c not in (reader.fieldnames or [])]
                if missing:
                    error_msg = f"Missing columns: {missing}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "processed": 0}

                docs: List[Document] = []
                ids: List[str] = []
                batch_size = 100

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

                        doc = self._doc_from_row(row, source_filename)
                        docs.append(doc)
                        ids.append(self._id_for(alias, source_filename))

                        # Process in batches
                        if len(docs) >= batch_size:
                            try:
                                self.vs.add_documents(docs, ids=ids)
                                processed += len(docs)
                                docs.clear()
                                ids.clear()
                                logger.info(f"Processed {processed} records...")
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
                        self.vs.add_documents(docs, ids=ids)
                        processed += len(docs)
                    except Exception as final_error:
                        logger.error(f"Final batch error: {str(final_error)}")
                        errors.append(f"Final batch error: {str(final_error)}")

            logger.info(f"✅ Ingestion completed - Processed: {processed}, Skipped: {skipped}")
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

    def search_fields(self, query: str, limit: int = 5, relevance_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """
        Vector similarity search with relevance filtering.
        
        Args:
            query: Search query
            limit: Maximum number of results
            relevance_threshold: Maximum distance (lower = more similar, default 1.0)
        """
        logger.info(f"🔍 Searching fields for: '{query}'")
        try:
            hits_with_scores = self.vs.similarity_search_with_score(query, k=limit)
            
            results = []

            for doc, distance in hits_with_scores:
                # Filter by relevance (smaller distance = more similar)
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

    def get_field_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """Get field by exact alias match."""
        logger.info(f"🔍 Getting field by alias: {alias}")
        try:
            hits = self.vs.similarity_search(
                query=alias,
                k=1,
                filter={"alias": {"$eq": alias}},
            )
            if not hits:
                return None
                
            doc = hits[0]
            md = doc.metadata or {}
            return {
                "alias": md.get("alias"),
                "old_alias": md.get("old_alias"),
                "long_name": md.get("long_name"),
                "definition": md.get("definition"),
                "data_type": md.get("data_type"),
                "example_value": md.get("example_value"),
                "remark": md.get("remark"),
                "source_file": md.get("source_file"),
                "ingested_at": md.get("ingested_at"),
            }
        except Exception as e:
            logger.error(f"❌ Error getting field by alias: {str(e)}")
            return None

    def get_all_fields(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """List all fields in the collection."""
        logger.info(f"📋 Getting all fields (limit: {limit})")
        try:
            sql = text("""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :col
                LIMIT :lim
            """)
            
            results = []
            with self.engine.begin() as conn:
                for doc, metadata in conn.execute(sql, {"col": self.COLLECTION, "lim": limit}):
                    md = metadata or {}
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
                    })
                    
            logger.info(f"✅ Retrieved {len(results)} fields")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting fields: {str(e)}")
            return []

    def delete_by_source_file(self, source_file: str) -> Dict[str, Any]:
        """Delete all records from a specific source file."""
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
                result = conn.execute(sql, {"col": self.COLLECTION, "src": source_file})
                deleted = result.rowcount or 0
                
            logger.info(f"✅ Deleted {deleted} fields from {source_file}")
            return {"success": True, "deleted_count": deleted}
            
        except Exception as e:
            error_msg = f"Error deleting fields: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}