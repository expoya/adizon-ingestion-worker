"""
LangGraph Ingestion Workflow for document processing.

This workflow handles:
1. Loading documents from MinIO (PDF, DOCX, TXT)
2. Splitting into chunks
3. Storing embeddings in PGVector
4. Extracting entities for Neo4j via LLM
5. Callback to backend to update document status
"""

import json
import os
import re
import tempfile
from typing import List, TypedDict

import httpx
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from core.config import get_settings
from services.graph_store import get_graph_store_service
from services.schema_factory import get_schema_factory
from services.storage import get_minio_service
from services.vector_store import get_vector_store_service

settings = get_settings()


class IngestionState(TypedDict):
    """State for the ingestion workflow."""

    document_id: str
    storage_path: str
    filename: str
    callback_url: str
    text_chunks: List[Document]
    entities: List[dict]
    relationships: List[dict]
    vector_ids: List[str]
    error: str | None
    status: str


def sanitize_text(text: str) -> str:
    """
    Sanitize text for PostgreSQL compatibility.

    Removes:
    - NUL bytes (0x00) which PostgreSQL text fields cannot contain
    - Other problematic control characters
    """
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def sanitize_documents(documents: List[Document]) -> List[Document]:
    """Sanitize all documents' page_content for PostgreSQL compatibility."""
    for doc in documents:
        doc.page_content = sanitize_text(doc.page_content)
    return documents


def get_llm() -> ChatOpenAI:
    """Get the configured LLM for graph extraction."""
    return ChatOpenAI(
        openai_api_base=settings.embedding_api_url,
        openai_api_key=settings.embedding_api_key,
        model_name=settings.llm_model_name,
        temperature=0,
    )


async def load_node(state: IngestionState) -> dict:
    """Load document from MinIO and extract text."""
    try:
        minio = get_minio_service()
        content = await minio.download_file(state["storage_path"])

        filename_lower = state["filename"].lower()
        documents: List[Document] = []
        tmp_path: str | None = None

        try:
            if filename_lower.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                print(f"   📄 Processing PDF with Unstructured: {state['filename']}")
                loader = UnstructuredFileLoader(tmp_path, mode="elements", strategy="fast")
                documents = loader.load()

            elif filename_lower.endswith(".docx"):
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                loader = Docx2txtLoader(tmp_path)
                documents = loader.load()

            elif filename_lower.endswith(".txt") or filename_lower.endswith(".md"):
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='wb') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                loader = TextLoader(tmp_path, encoding="utf-8")
                try:
                    documents = loader.load()
                except UnicodeDecodeError:
                    loader = TextLoader(tmp_path, encoding="latin-1")
                    documents = loader.load()

            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1")

                documents = [
                    Document(
                        page_content=text,
                        metadata={
                            "source": state["storage_path"],
                            "filename": state["filename"],
                        },
                    )
                ]

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        for doc in documents:
            doc.metadata["document_id"] = state["document_id"]
            doc.metadata["filename"] = state["filename"]

        documents = sanitize_documents(documents)

        return {
            "text_chunks": documents,
            "status": "loaded",
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"❌ ERROR in load_node:")
        print(f"   {str(e)}")
        print(f"\n{error_trace}")
        print(f"{'='*60}\n")
        return {
            "error": f"Failed to load document: {str(e)}",
            "traceback": error_trace,
            "status": "error",
        }


async def split_node(state: IngestionState) -> dict:
    """Split loaded documents into smaller chunks."""
    if state.get("error"):
        return {}

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[Document] = []
        for doc in state["text_chunks"]:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.page_content = sanitize_text(chunk.page_content)

        return {
            "text_chunks": all_chunks,
            "status": "split",
        }

    except Exception as e:
        return {
            "error": f"Failed to split document: {str(e)}",
            "status": "error",
        }


async def vector_node(state: IngestionState) -> dict:
    """Store document chunks in PGVector."""
    if state.get("error"):
        return {}

    try:
        vector_store = get_vector_store_service()
        ids = await vector_store.add_documents(
            chunks=state["text_chunks"],
            document_id=state["document_id"],
        )

        return {
            "vector_ids": ids,
            "status": "vectorized",
        }

    except Exception as e:
        return {
            "error": f"Failed to store vectors: {str(e)}",
            "status": "error",
        }


async def graph_node(state: IngestionState) -> dict:
    """
    Extract entities and relationships using LLM and store in Neo4j.

    Uses dynamic ontology-based structured output via SchemaFactory.
    """
    if state.get("error"):
        return {}

    entities: List[dict] = []
    relationships: List[dict] = []

    try:
        schema_factory = get_schema_factory()
        ontology_config = schema_factory.load_config()
        models = schema_factory.get_dynamic_models()
        system_instruction = schema_factory.get_system_instruction()

        ExtractionResult = models["ExtractionResult"]

        print(f"   📋 Ontology loaded: {ontology_config.domain_name}")
        print(f"      - Node types: {', '.join(schema_factory.get_node_types())}")
        print(f"      - Relationship types: {', '.join(schema_factory.get_relationship_types())}")

        llm = get_llm()
        structured_llm = llm.with_structured_output(ExtractionResult)

        max_chunks_for_graph = 5
        chunks_to_process = state["text_chunks"][:max_chunks_for_graph]

        if not chunks_to_process:
            print("   ⚠️ No chunks to process for graph extraction")
            return {
                "entities": [],
                "relationships": [],
                "status": "graph_skipped",
            }

        print(f"   🔍 Extracting graph from {len(chunks_to_process)} chunks using {settings.llm_model_name}...")

        seen_entities: set = set()

        for i, chunk in enumerate(chunks_to_process):
            try:
                extraction_prompt = f"""{system_instruction}

## Text to Analyze
Extract all entities and relationships from the following text:

---
{chunk.page_content}
---

Return the extracted nodes and relationships in the specified JSON format.
"""
                result = structured_llm.invoke(extraction_prompt)

                for node in result.nodes:
                    entity_key = f"{node.type}:{node.name}"
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities.append({
                            "label": node.type,
                            "name": node.name,
                            "properties": node.properties or {},
                        })

                for rel in result.relationships:
                    relationships.append({
                        "from_label": rel.source_type,
                        "from_name": rel.source_name,
                        "to_label": rel.target_type,
                        "to_name": rel.target_name,
                        "type": rel.type,
                        "properties": rel.properties or {},
                    })

            except Exception as chunk_error:
                print(f"   ⚠️ Chunk {i+1} extraction failed: {chunk_error}")
                continue

        if entities or relationships:
            graph_store = get_graph_store_service()
            result = await graph_store.add_graph_documents(
                entities=entities,
                relationships=relationships,
                document_id=state["document_id"],
                source_file=state.get("filename"),
            )
            print(f"   ✓ Graph extracted: {result['nodes_created']} nodes (PENDING), {result['relationships_created']} relationships (PENDING)")
        else:
            print("   ⚠️ No entities extracted from document")

        return {
            "entities": entities,
            "relationships": relationships,
            "status": "graph_extracted",
        }

    except FileNotFoundError as e:
        print(f"   ⚠️ Ontology config not found: {e}")
        return {
            "entities": [],
            "relationships": [],
            "status": "graph_skipped",
        }

    except Exception as e:
        print(f"   ⚠️ Graph extraction failed (non-fatal): {e}")
        return {
            "entities": [],
            "relationships": [],
            "status": "graph_skipped",
        }


async def finalize_node(state: IngestionState) -> dict:
    """
    Callback to backend to update document status.
    """
    try:
        if state.get("error"):
            new_status = "ERROR"
            error_msg = state["error"]
            # --- CRITICAL: Fehler explizit auf Konsole ausgeben ---
            print(f"\n{'='*60}")
            print(f"❌ CRITICAL ERROR in workflow:")
            print(f"   Error: {error_msg}")
            if "traceback" in state:
                print(f"   Traceback: {state['traceback']}")
            print(f"   Document: {state.get('filename', 'unknown')}")
            print(f"   Document ID: {state.get('document_id', 'unknown')}")
            print(f"{'='*60}\n")
            # -------------------------------------------------------
        else:
            new_status = "INDEXED"
            error_msg = None

        # Call backend to update document status using dynamic callback URL
        callback_url = state.get("callback_url")
        if callback_url:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    callback_url,
                    json={
                        "status": new_status,
                        "error_message": error_msg,
                    },
                    timeout=30.0,
                )
                if response.status_code != 200:
                    print(f"   ⚠️ Failed to update backend status: {response.status_code}")
                else:
                    print(f"   ✓ Document status updated: {new_status}")
        else:
            print(f"   ⚠️ No callback URL provided, skipping status update")

        return {
            "status": new_status.lower(),
        }

    except Exception as e:
        print(f"Error finalizing document: {e}")
        return {
            "error": f"Failed to finalize: {str(e)}",
            "status": "error",
        }


def should_continue(state: IngestionState) -> str:
    """Determine if workflow should continue or handle error."""
    if state.get("error"):
        return "finalize"
    return "continue"


def create_ingestion_graph() -> StateGraph:
    """Create and compile the ingestion workflow graph."""

    workflow = StateGraph(IngestionState)

    workflow.add_node("load", load_node)
    workflow.add_node("split", split_node)
    workflow.add_node("vector", vector_node)
    workflow.add_node("graph", graph_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("load")

    workflow.add_conditional_edges(
        "load",
        should_continue,
        {"continue": "split", "finalize": "finalize"},
    )

    workflow.add_conditional_edges(
        "split",
        should_continue,
        {"continue": "vector", "finalize": "finalize"},
    )

    workflow.add_conditional_edges(
        "vector",
        should_continue,
        {"continue": "graph", "finalize": "finalize"},
    )

    workflow.add_edge("graph", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Compiled graph instance
ingestion_graph = create_ingestion_graph()


async def run_ingestion_workflow(
    document_id: str,
    storage_path: str,
    filename: str,
    callback_url: str,
) -> dict:
    """
    Run the ingestion workflow for a document.

    Args:
        document_id: UUID of the document
        storage_path: Path to the document in MinIO
        filename: Original filename
        callback_url: Webhook URL for status updates

    Returns:
        Final workflow state
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting ingestion workflow for: {filename}")
    print(f"   Document ID: {document_id}")
    print(f"   Storage path: {storage_path}")
    print(f"   Callback URL: {callback_url}")
    print(f"{'='*60}\n")

    initial_state: IngestionState = {
        "document_id": document_id,
        "storage_path": storage_path,
        "filename": filename,
        "callback_url": callback_url,
        "text_chunks": [],
        "entities": [],
        "relationships": [],
        "vector_ids": [],
        "error": None,
        "status": "starting",
    }

    result = await ingestion_graph.ainvoke(initial_state)

    print(f"\n{'='*60}")
    print(f"✅ Workflow completed for: {filename}")
    print(f"   Final status: {result.get('status', 'unknown')}")
    print(f"{'='*60}\n")

    return dict(result)
