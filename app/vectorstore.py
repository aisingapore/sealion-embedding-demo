from __future__ import annotations

import fnmatch
import logging
from typing import Any

import chromadb
from chromadb import Collection

from app.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.HttpClient | None = None
_collection: Collection | None = None


def get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        try:
            _client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        except Exception:
            logger.error("Failed to connect to ChromaDB", exc_info=True)
            raise RuntimeError("Could not connect to ChromaDB.")
    return _client


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        try:
            client = get_client()
            _collection = client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            logger.error("Failed to get or create ChromaDB collection", exc_info=True)
            raise RuntimeError("Could not access the vector store collection.")
    return _collection


def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    chunk_indices: list[int],
    last_modified: float,
) -> None:
    try:
        collection = get_collection()
        ids = [f"{source}::{i}" for i in chunk_indices]
        metadatas = [
            {
                "source": source,
                "chunk_index": i,
                "last_modified": last_modified,
            }
            for i in chunk_indices
        ]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception:
        logger.error("Failed to upsert chunks for '%s'", source, exc_info=True)
        raise RuntimeError(f"Failed to store chunks for '{source}'.")


def delete_document(source: str) -> int:
    try:
        collection = get_collection()
        if any(c in source for c in ("*", "?", "[", "]")):
            all_results = collection.get(include=["metadatas"])
            sources = {m["source"] for m in all_results["metadatas"]}
            matched = [s for s in sources if fnmatch.fnmatch(s, source)]
            total = 0
            for s in matched:
                results = collection.get(where={"source": s})
                if results["ids"]:
                    collection.delete(ids=results["ids"])
                    logger.info(f"Deleted {len(results['ids'])} chunks for '{s}'")
                    total += len(results["ids"])
            return total
        results = collection.get(where={"source": source})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for '{source}'")
            return len(results["ids"])
        return 0
    except Exception:
        logger.error("Failed to delete document '%s'", source, exc_info=True)
        raise RuntimeError(f"Failed to delete '{source}' from the vector store.")


def search(
    query_embedding: list[float],
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        where = {"source": source_filter} if source_filter else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite. Convert to similarity.
            score = 1 - distance / 2
            hits.append(
                {
                    "id": doc_id,
                    "score": round(score, 4),
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
        return hits
    except Exception:
        logger.error("Vector store search failed", exc_info=True)
        raise RuntimeError("Search failed. Please try again.")


def list_documents() -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        if not results["ids"]:
            return []

        docs: dict[str, dict] = {}
        for meta in results["metadatas"]:
            source = meta["source"]
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "last_modified": meta.get("last_modified", 0),
                }
            docs[source]["chunk_count"] += 1

        return list(docs.values())
    except Exception:
        logger.error("Failed to list documents from vector store", exc_info=True)
        raise RuntimeError("Failed to retrieve document list.")


def get_document_last_modified(source: str) -> float | None:
    try:
        collection = get_collection()
        results = collection.get(
            where={"source": source},
            include=["metadatas"],
            limit=1,
        )
        if results["ids"]:
            return results["metadatas"][0].get("last_modified")
        return None
    except Exception:
        logger.warning(
            "Could not retrieve last_modified for '%s', will re-index",
            source,
            exc_info=True,
        )
        return None
