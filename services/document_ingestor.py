import os
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

from sqlalchemy import create_engine, text

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector  # langchain-postgres >= 0.0.15
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from config import get_settings, logger


# -------------------------
# Vector Store Service
# -------------------------
class DocumentIngestor:
    """
    Stores field definitions in a Postgres+pgvector collection via LangChain.
    Provides ingest/search/get/delete operations analogous to the original service.
    """

    COLLECTION = "field_definitions"  # logical collection name inside langchain_pg_* tables

    def __init__(self, embeddings: Optional[OpenAIEmbeddings] = None):
        logger.info("🔧 Initializing Vector Field Store...")

        self.settings = get_settings()

        # IMPORTANT: psycopg3 driver is required; `+psycopg2` won't work with langchain-postgres
        self.connection_url = (
            f"postgresql+psycopg://{self.settings.DATABASE_USER}:"
            f"{self.settings.DATABASE_PASSWORD}@{self.settings.DATABASE_HOST}:"
            f"{self.settings.DATABASE_PORT}/{self.settings.DATABASE_NAME}"
        )

        # Embeddings (swap to your provider if desired)
        # e.g. OpenAI text-embedding-3-large; requires OPENAI_API_KEY
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-large")

        # LangChain PGVector store (creates langchain_pg_collection / langchain_pg_embedding if missing)
        self.vs = PGVector(
            embeddings=self.embeddings,
            collection_name=self.COLLECTION,
            connection=self.connection_url,
            use_jsonb=True,         # JSONB metadata; supports $-ops filtering
            create_extension=True,  # will CREATE EXTENSION pgvector if permitted
        )

        # Raw SQL access for listing/deletion by metadata (PGVector delete() only accepts ids)
        self.engine = create_engine(self.connection_url)

        logger.info("✅ Vector Field Store initialized")

    # -------------------------
    # Helpers
    # -------------------------
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
        """Build a LangChain Document with sensible page_content and searchable metadata."""
        alias = (row.get("Alias") or "").strip()
        long_name = (row.get("Long Name") or "").strip()
        definition = (row.get("Definition") or "").strip()
        data_type = (row.get("Data Type") or "").strip()
        example = (row.get("Example Value") or "").strip()
        remark = (row.get("Remark") or "").strip()
        old_alias = (row.get("Old Alias") or "").strip()

        # Human/LLM-friendly content (what gets embedded)
        content_lines = [
            f"Alias: {alias}",
            f"Long Name: {long_name}" if long_name else "",
            f"Definition: {definition}" if definition else "",
            f"Data Type: {data_type}" if data_type else "",
            f"Example: {example}" if example else "",
            f"Remark: {remark}" if remark else "",
        ]
        page_content = "\n".join([ln for ln in content_lines if ln])

        # Rich metadata (query-filterable)
        metadata = {
            "alias": alias,
            "old_alias": old_alias or None,
            "long_name": long_name or None,
            "data_type": data_type or None,
            "remark": remark or None,
            "source_file": source_filename,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _id_for(alias: str, source_filename: str) -> str:
        """Deterministic ID; ensures updates overwrite rather than duplicate."""
        return f"{source_filename}::{alias.lower()}"

    # -------------------------
    # Public API (mirrors your old service)
    # -------------------------
    def ingest_field_definitions(self, file_path: str) -> Dict[str, Any]:
        logger.info(f"📖 Starting ingestion of file: {file_path}")

        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": 0}

        processed = 0
        skipped = 0
        errors: List[str] = []
        source_filename = Path(file_path).name
        docs: List[Document] = []
        ids: List[str] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Validate header
                missing = [c for c in self._expected_columns() if c not in (reader.fieldnames or [])]
                if missing:
                    error_msg = f"Missing required columns. Missing: {missing}, Found: {reader.fieldnames}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "processed": 0}

                for row_num, row in enumerate(reader, start=2):
                    try:
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
                        processed += 1

                        # Optional: batch flush every 500 rows to keep memory modest
                        if len(docs) >= 500:
                            self.vs.add_documents(docs, ids=ids)
                            docs.clear()
                            ids.clear()
                            logger.info(f"Processed {processed} records so far...")

                    except Exception as e:
                        msg = f"Row {row_num}: {str(e)}"
                        logger.error(msg)
                        errors.append(msg)
                        skipped += 1

            # Final add
            if docs:
                self.vs.add_documents(docs, ids=ids)

            logger.info(
                f"✅ Ingestion completed - Processed: {processed}, Skipped: {skipped}, Errors: {len(errors)}"
            )
            return {
                "success": True,
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "source_file": source_filename,
            }

        except Exception as e:
            error_msg = f"Error reading/ingesting {file_path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "processed": processed}

    def search_fields(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Vector similarity search across alias/long_name/definition (embedded content),
        with metadata returned for each hit.
        """
        logger.info(f"🔍 Searching fields for: '{query}'")
        try:
            hits = self.vs.similarity_search(query, k=limit)
            out = []
            for d in hits:
                md = d.metadata or {}
                out.append(
                    {
                        "alias": md.get("alias"),
                        "old_alias": md.get("old_alias"),
                        "long_name": md.get("long_name"),
                        "definition": _extract_section(d.page_content, "Definition"),
                        "data_type": md.get("data_type"),
                        "example_value": _extract_section(d.page_content, "Example"),
                        "remark": md.get("remark"),
                        "source_file": md.get("source_file"),
                        "ingested_at": md.get("ingested_at"),
                    }
                )
            logger.info(f"✅ Found {len(out)} matching fields")
            return out
        except Exception as e:
            logger.error(f"❌ Error searching fields: {str(e)}")
            return []

    def get_field_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """
        Fetch by exact alias (uses metadata filter). We send alias as the query to ensure
        a stable result even if multiple docs share the alias across files.
        """
        logger.info(f"🔍 Getting field by alias: {alias}")
        try:
            hits = self.vs.similarity_search(
                query=alias,
                k=1,
                # JSONB filters support $eq/$in/etc.; plain equality is also supported.
                filter={"alias": {"$eq": alias}},
            )
            if not hits:
                return None
            d = hits[0]
            md = d.metadata or {}
            return {
                "alias": md.get("alias"),
                "old_alias": md.get("old_alias"),
                "long_name": md.get("long_name"),
                "definition": _extract_section(d.page_content, "Definition"),
                "data_type": md.get("data_type"),
                "example_value": _extract_section(d.page_content, "Example"),
                "remark": md.get("remark"),
                "source_file": md.get("source_file"),
                "ingested_at": md.get("ingested_at"),
            }
        except Exception as e:
            logger.error(f"❌ Error getting field by alias: {str(e)}")
            return None

    def get_all_fields(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        List documents in this collection directly from langchain_pg_embedding.
        (LangChain’s delete/list APIs are id-focused; raw SQL is simplest to “list all”.)
        """
        logger.info(f"📋 Getting all fields (limit: {limit})")
        try:
            sql = text(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :col
                LIMIT :lim
                """
            )
            rows = []
            with self.engine.begin() as conn:
                for doc, cmetadata in conn.execute(sql, {"col": self.COLLECTION, "lim": limit}):
                    rows.append(_row_from_document_and_meta(doc, cmetadata))
            logger.info(f"✅ Retrieved {len(rows)} fields")
            return rows
        except Exception as e:
            logger.error(f"❌ Error getting fields: {str(e)}")
            return []

    def delete_by_source_file(self, source_file: str) -> Dict[str, Any]:
        """
        Delete all vectors whose metadata.source_file == source_file for this collection.
        Uses SQL since PGVector.delete() currently deletes by ids.
        """
        logger.info(f"🗑️ Deleting fields from source file: {source_file}")
        try:
            sql = text(
                """
                DELETE FROM langchain_pg_embedding e
                USING langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name = :col
                  AND e.cmetadata->>'source_file' = :src
                """
            )
            with self.engine.begin() as conn:
                result = conn.execute(sql, {"col": self.COLLECTION, "src": source_file})
                deleted = result.rowcount or 0
            logger.info(f"✅ Deleted {deleted} fields from {source_file}")
            return {"success": True, "deleted_count": deleted}
        except Exception as e:
            error_msg = f"Error deleting fields: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


# -------------------------
# Small LangGraph wrapper
# -------------------------
class IngestSearchState(TypedDict, total=False):
    action: Literal["ingest", "search", "delete_by_source"]
    file_path: str
    query: str
    limit: int
    source_file: str
    result: Any
    error: str


def build_graph(store: DocumentIngestor):
    """
    LangGraph app that routes to: ingest, search, delete_by_source.
    Use:
        app = build_graph(DocumentIngestor())
        app.invoke({"action": "ingest", "file_path": "fields.csv"})
        app.invoke({"action": "search", "query": "transaction amount", "limit": 10})
        app.invoke({"action": "delete_by_source", "source_file": "fields.csv"})
    """
    def route(state: IngestSearchState):
        return state["action"]

    def do_ingest(state: IngestSearchState):
        res = store.ingest_field_definitions(state["file_path"])
        return {"result": res}

    def do_search(state: IngestSearchState):
        res = store.search_fields(state["query"], limit=state.get("limit", 50))
        return {"result": res}

    def do_delete(state: IngestSearchState):
        res = store.delete_by_source_file(state["source_file"])
        return {"result": res}

    graph = StateGraph(IngestSearchState)
    graph.add_node("ingest", do_ingest)
    graph.add_node("search", do_search)
    graph.add_node("delete_by_source", do_delete)

    graph.add_conditional_edges(START, route, {
        "ingest": "ingest",
        "search": "search",
        "delete_by_source": "delete_by_source",
    })
    graph.add_edge("ingest", END)
    graph.add_edge("search", END)
    graph.add_edge("delete_by_source", END)

    return graph.compile()


# -------------------------
# Utility parse helpers
# -------------------------
def _extract_section(page: str, label: str) -> Optional[str]:
    """
    Given the page_content we create (key: value lines), pull the section.
    """
    prefix = f"{label}: "
    for line in page.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip() or None
    return None


def _row_from_document_and_meta(document_str: str, meta_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alias": (meta_json or {}).get("alias"),
        "old_alias": (meta_json or {}).get("old_alias"),
        "long_name": (meta_json or {}).get("long_name"),
        "definition": _extract_section(document_str or "", "Definition"),
        "data_type": (meta_json or {}).get("data_type"),
        "example_value": _extract_section(document_str or "", "Example"),
        "remark": (meta_json or {}).get("remark"),
        "source_file": (meta_json or {}).get("source_file"),
        "ingested_at": (meta_json or {}).get("ingested_at"),
    }
