# ============================================================
# Vector Pipeline — ChromaDB embeddings & semantic memory
# Used by Memory Agent & Orchestrator Agent
# ============================================================

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import json
import hashlib
import os


# ── 1. Init ChromaDB Client ──────────────────────────────────

def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.Client:
    """Initialise a persistent local ChromaDB client."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info(f"ChromaDB client initialised | Path: {persist_dir}")
    return client


# ── 2. Load Embedding Model ──────────────────────────────────

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a free sentence-transformers embedding model."""
    model = SentenceTransformer(model_name)
    logger.info(f"Embedding model loaded: {model_name}")
    return model


# ── 3. Get or Create Collection ──────────────────────────────

def get_or_create_collection(
    client: chromadb.Client,
    collection_name: str = "ds_crew_memory",
) -> chromadb.Collection:
    """Get existing or create new ChromaDB collection."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection ready: {collection_name} | "
                f"Documents: {collection.count()}")
    return collection


# ── 4. Generate Document ID ──────────────────────────────────

def _generate_id(content: str) -> str:
    """Generate a stable unique ID from content hash."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ── 5. Store EDA Summary ─────────────────────────────────────

def store_eda_summary(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    eda_summary: dict,
    dataset_name: str = "dataset",
) -> str:
    """
    Embed and store EDA summary findings into ChromaDB.
    Returns the document ID.
    """
    # Build a human-readable text representation for embedding
    basic   = eda_summary.get("basic_summary", {})
    shape   = basic.get("shape", {})
    nulls   = basic.get("null_counts", {})
    num_col = basic.get("numeric_columns", [])
    cat_col = basic.get("categorical_columns", [])

    text = f"""
    Dataset: {dataset_name}
    Shape: {shape.get('rows', 0)} rows x {shape.get('columns', 0)} columns
    Numeric columns: {', '.join(num_col)}
    Categorical columns: {', '.join(cat_col)}
    Total missing values: {sum(nulls.values())}
    Duplicate rows: {basic.get('duplicate_rows', 0)}
    Memory usage: {basic.get('memory_mb', 0)} MB
    """.strip()

    doc_id    = _generate_id(f"eda_{dataset_name}")
    embedding = embedding_model.encode(text).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "type":         "eda_summary",
            "dataset":      dataset_name,
            "timestamp":    datetime.now().isoformat(),
            "num_rows":     shape.get("rows", 0),
            "num_columns":  shape.get("columns", 0),
        }],
    )

    logger.success(f"EDA summary stored in ChromaDB | ID: {doc_id}")
    return doc_id


# ── 6. Store Model Results ───────────────────────────────────

def store_model_results(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    automl_results: dict,
) -> str:
    """Embed and store AutoML results into ChromaDB."""
    best_name    = automl_results.get("best_model_name", "unknown")
    best_metrics = automl_results.get("best_metrics", {})
    task_type    = automl_results.get("task_type", "unknown")
    all_results  = automl_results.get("all_results", {})

    # Summarise all model scores
    model_scores = []
    for name, metrics in all_results.items():
        if "error" not in metrics:
            primary = "accuracy" if task_type == "classification" else "r2"
            score   = metrics.get(primary, "N/A")
            model_scores.append(f"{name}: {score}")

    text = f"""
    Task type: {task_type}
    Best model: {best_name}
    Best model metrics: {json.dumps(best_metrics)}
    All model scores: {', '.join(model_scores)}
    Target column: {automl_results.get('target_column', 'unknown')}
    Train shape: {automl_results.get('train_shape', 'unknown')}
    Test shape: {automl_results.get('test_shape', 'unknown')}
    """.strip()

    doc_id    = _generate_id(f"model_{best_name}_{datetime.now().isoformat()}")
    embedding = embedding_model.encode(text).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "type":       "model_results",
            "task_type":  task_type,
            "best_model": best_name,
            "timestamp":  datetime.now().isoformat(),
        }],
    )

    logger.success(f"Model results stored in ChromaDB | ID: {doc_id}")
    return doc_id


# ── 7. Store Agent Insight ───────────────────────────────────

def store_agent_insight(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    agent_name: str,
    insight: str,
    metadata: dict = None,
) -> str:
    """
    Store a free-text insight from any agent into ChromaDB.
    Enables agents to retrieve past context semantically.
    """
    doc_id    = _generate_id(f"{agent_name}_{insight[:50]}_{datetime.now().isoformat()}")
    embedding = embedding_model.encode(insight).tolist()

    meta = {
        "type":       "agent_insight",
        "agent":      agent_name,
        "timestamp":  datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[insight],
        metadatas=[meta],
    )

    logger.info(f"Agent insight stored | Agent: {agent_name} | ID: {doc_id}")
    return doc_id


# ── 8. Semantic Search ───────────────────────────────────────

def semantic_search(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    query: str,
    n_results: int = 5,
    filter_type: str = None,
) -> list[dict]:
    """
    Search ChromaDB for documents semantically similar to query.
    Optionally filter by document type.
    Returns list of result dicts.
    """
    query_embedding = embedding_model.encode(query).tolist()

    where_clause = {"type": filter_type} if filter_type else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count() or 1),
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    for i, doc in enumerate(results["documents"][0]):
        formatted.append({
            "document": doc,
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
        })

    logger.info(f"Semantic search | Query: '{query[:50]}...' | "
                f"Results: {len(formatted)}")
    return formatted


# ── 9. Store DataFrame Sample ────────────────────────────────

def store_dataframe_sample(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    sample_size: int = 5,
) -> str:
    """Store a text description of the DataFrame sample for context retrieval."""
    sample  = df.head(sample_size).to_string()
    dtypes  = df.dtypes.to_string()

    text = f"""
    Dataset name: {dataset_name}
    Columns and types:
    {dtypes}

    Sample rows:
    {sample}
    """.strip()

    doc_id    = _generate_id(f"df_sample_{dataset_name}")
    embedding = embedding_model.encode(text).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "type":      "dataframe_sample",
            "dataset":   dataset_name,
            "rows":      df.shape[0],
            "columns":   df.shape[1],
            "timestamp": datetime.now().isoformat(),
        }],
    )

    logger.success(f"DataFrame sample stored | Dataset: {dataset_name}")
    return doc_id


# ── 10. Get Full Memory Context ──────────────────────────────

def get_full_context(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    query: str = "dataset analysis summary",
    n_results: int = 10,
) -> str:
    """
    Retrieve all relevant memory as a single context string.
    Used by Orchestrator & Reporting agents.
    """
    results = semantic_search(collection, embedding_model, query, n_results)

    if not results:
        return "No prior context found in vector memory."

    context_parts = []
    for r in results:
        meta = r["metadata"]
        context_parts.append(
            f"[{meta.get('type', 'unknown')} | {meta.get('timestamp', '')}]\n"
            f"{r['document']}"
        )

    full_context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Full context retrieved | {len(results)} documents")
    return full_context


# ── 11. Init Full Vector Pipeline ────────────────────────────

def init_vector_pipeline(
    persist_dir: str = "./chroma_db",
    collection_name: str = "ds_crew_memory",
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> tuple:
    """
    One-call setup for the full vector pipeline.
    Returns (client, collection, embedding_model)
    """
    client          = get_chroma_client(persist_dir)
    collection      = get_or_create_collection(client, collection_name)
    embedding_model = get_embedding_model(embedding_model_name)

    logger.success("Vector pipeline initialised")
    return client, collection, embedding_model