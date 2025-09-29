from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from services.document_ingestor import DocumentIngestor
from dependencies import get_ingestor
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import os
from fastapi import Query

router = APIRouter()

@router.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...), ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Ingest a CSV file containing field definitions.
    """
    # Save to a temp file first
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingestor.ingest_field_definitions(tmp_path)
    finally:
        os.remove(tmp_path)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/ingest-examples")
async def ingest_query_examples(
    file: UploadFile = File(...), 
    ingestor: DocumentIngestor = Depends(get_ingestor)
):
    """
    Ingest a CSV file containing query examples.
    Expected columns: Description, Query, Category (optional), Complexity (optional), Fields_Used (optional)
    """
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingestor.ingest_query_examples(tmp_path)
    finally:
        os.remove(tmp_path)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.get("/search-fields")
def search_fields(q: str = Query(..., description="Search text"), limit: int = 50, ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Search field definitions by alias, name, or definition.
    """
    results = ingestor.search_fields(q, limit)
    return {"results": results, "count": len(results)}


@router.get("/fields")
def get_all_fields(limit: int = 1000, ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Get all field definitions.
    """
    results = ingestor.get_all_fields(limit)
    return {"results": results, "count": len(results)}


@router.get("/fields/{alias}")
def get_field_by_alias(alias: str, ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Get a single field definition by alias.
    """
    result = ingestor.get_field_by_alias(alias)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Field not found"})
    return result


@router.delete("/fields/source/{source_file}")
def delete_by_source_file(source_file: str, ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Delete all field definitions ingested from a specific source file.
    """
    result = ingestor.delete_fields_by_source_file(source_file)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.delete("/examples/source/{source_file}")
def delete_examples_by_source_file(source_file: str, ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Delete all query examples ingested from a specific source file.
    """
    result = ingestor.delete_examples_by_source_file(source_file)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@router.delete("/clear-all-collections")
def clear_all_collections(ingestor: DocumentIngestor = Depends(get_ingestor)):
    """
    Clear all collections.
    """
    result = ingestor.clear_all_collections()
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result