"""
services/rag_service.py

Production RAG Service

Responsibilities
----------------
- Vector database management
- Document indexing
- Semantic retrieval
- Metadata filtering
- Context construction
- Source tracking
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings

from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RagService:
    """
    Production Retrieval-Augmented Generation (RAG) service.

    This service orchestrates:
    - document indexing
    - vector storage
    - semantic retrieval
    """

    DEFAULT_COLLECTION = "documents"

    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = DEFAULT_COLLECTION,
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self.embedding_service = EmbeddingService()

        self.collection = self._get_or_create_collection()

        logger.info(
            "RAG Service initialized successfully."
        )

    # =====================================================
    # Collection
    # =====================================================

    def _get_or_create_collection(self):

        try:

            collection = self.client.get_collection(
                self.collection_name
            )

            logger.info(
                "Loaded collection '%s'",
                self.collection_name,
            )

            return collection

        except Exception:

            logger.info(
                "Creating collection '%s'",
                self.collection_name,
            )

            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "AI Memory RAG Collection"
                },
            )

    # =====================================================
    # Collection Info
    # =====================================================

    def collection_info(self) -> Dict:

        return {

            "name": self.collection_name,

            "count": self.collection.count(),

            "persist_directory":
                self.persist_directory,

        }

    # =====================================================
    # Health Check
    # =====================================================

    def health(self) -> Dict:

        return {

            "status": "healthy",

            "collection":
                self.collection_name,

            "documents":
                self.collection.count(),

            "embedding_model":
                self.embedding_service.model_name_info(),

        }

    # =====================================================
    # Reset Collection
    # =====================================================

    def reset_collection(self):

        logger.warning(
            "Resetting collection..."
        )

        self.client.delete_collection(
            self.collection_name
        )

        self.collection = (
            self.client.create_collection(
                name=self.collection_name
            )
        )

        logger.info("Collection reset.")

    # =====================================================
    # Add Chunks
    # =====================================================

    def add_chunks(
        self,
        chunks: List[Dict],
    ) -> int:
        """
        Add processed chunks into ChromaDB.

        Expected format:

        {
            "content": "...",
            "embedding": [...],
            "metadata": {...}
        }
        """

        if not chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:

            ids.append(
                str(uuid.uuid4())
            )

            documents.append(
                chunk["content"]
            )

            embeddings.append(
                chunk["embedding"].tolist()
            )

            metadatas.append(
                chunk["metadata"]
            )

        self.collection.add(

            ids=ids,

            documents=documents,

            embeddings=embeddings,

            metadatas=metadatas,

        )

        logger.info(
            "Indexed %d chunks.",
            len(chunks),
        )

        return len(chunks)

    # =====================================================
    # Index Processed Document
    # =====================================================

    def index_document(
        self,
        processed_document: Dict,
    ) -> int:
        """
        Index output generated by
        DocumentService.process_document()
        """

        chunks = []

        for content, metadata, embedding in zip(

            processed_document["chunks"],

            processed_document["chunk_metadata"],

            processed_document["embeddings"],

        ):

            chunks.append(

                {

                    "content": content,

                    "embedding": embedding,

                    "metadata": metadata,

                }

            )

        return self.add_chunks(chunks)

    # =====================================================
    # Add Raw Documents
    # =====================================================

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> int:
        """
        Index raw documents directly.
        """

        embeddings = self.embedding_service.embed_documents(
            documents
        )

        ids = [
            str(uuid.uuid4())
            for _ in documents
        ]

        if metadata is None:

            metadata = [{} for _ in documents]

        self.collection.add(

            ids=ids,

            documents=documents,

            embeddings=embeddings.tolist(),

            metadatas=metadata,

        )

        logger.info(
            "%d raw documents indexed.",
            len(documents),
        )

        return len(documents)

    # =====================================================
    # Document Count
    # =====================================================

    def count(self) -> int:

        return self.collection.count()

      # =====================================================
    # Embed Query
    # =====================================================

    def _embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Convert query into embedding.
        """

        embedding = self.embedding_service.embed_query(query)

        return embedding.tolist()

    # =====================================================
    # Semantic Search
    # =====================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Perform semantic vector search.
        """

        query_embedding = self._embed_query(query)

        results = self.collection.query(

            query_embeddings=[query_embedding],

            n_results=top_k,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return self._format_results(results)

    # =====================================================
    # Similarity Search
    # =====================================================

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Alias for semantic search.
        """

        return self.semantic_search(
            query=query,
            top_k=top_k,
        )

    # =====================================================
    # Metadata Filter Search
    # =====================================================

    def search_with_filter(
        self,
        query: str,
        metadata_filter: Dict,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Semantic search using metadata filters.
        """

        embedding = self._embed_query(query)

        results = self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k,

            where=metadata_filter,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return self._format_results(results)

    # =====================================================
    # Search By Document ID
    # =====================================================

    def search_document(
        self,
        document_id: str,
    ) -> List[Dict]:
        """
        Retrieve all chunks for one document.
        """

        results = self.collection.get(

            where={

                "document_id": document_id

            },

            include=[

                "documents",

                "metadatas",

            ],
        )

        formatted = []

        documents = results.get("documents", [])

        metadata = results.get("metadatas", [])

        for doc, meta in zip(

            documents,

            metadata,

        ):

            formatted.append(

                {

                    "content": doc,

                    "metadata": meta,

                }

            )

        return formatted

    # =====================================================
    # Retrieve Context
    # =====================================================

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Retrieve context for PromptService.
        """

        results = self.semantic_search(

            query=query,

            top_k=top_k,

        )

        context = "\n\n".join(

            item["content"]

            for item in results

        )

        return {

            "context": context,

            "sources": [

                item["metadata"]

                for item in results

            ],

            "results": results,

        }

    # =====================================================
    # Build Context
    # =====================================================

    @staticmethod
    def build_context(
        results: List[Dict],
    ) -> str:
        """
        Convert retrieved chunks into
        a prompt-ready context.
        """

        context = []

        for index, item in enumerate(results, start=1):

            filename = item["metadata"].get(

                "filename",

                "Unknown",

            )

            context.append(

                f"[Document {index}]"

            )

            context.append(

                f"Source: {filename}"

            )

            context.append(

                item["content"]

            )

            context.append("")

        return "\n".join(context)

    # =====================================================
    # Internal Result Formatter
    # =====================================================

    @staticmethod
    def _format_results(
        results: Dict,
    ) -> List[Dict]:
        """
        Normalize ChromaDB output.
        """

        formatted = []

        documents = results.get(

            "documents",

            [[]],

        )[0]

        metadatas = results.get(

            "metadatas",

            [[]],

        )[0]

        distances = results.get(

            "distances",

            [[]],

        )[0]

        for document, metadata, distance in zip(

            documents,

            metadatas,

            distances,

        ):

            formatted.append(

                {

                    "content": document,

                    "metadata": metadata,

                    "distance": float(distance),

                    "score": 1 - float(distance),

                }

            )

        return formatted

    # =====================================================
    # Search API
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Main search API.
        """

        results = self.semantic_search(

            query=query,

            top_k=top_k,

        )

        return {

            "query": query,

            "count": len(results),

            "results": results,

            "context": self.build_context(

                results

            ),

        }

      # =====================================================
    # Load Cross Encoder
    # =====================================================

    def load_reranker(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Lazily initialize the reranker.
        """

        if hasattr(self, "_reranker"):
            return

        logger.info(
            "Loading CrossEncoder reranker..."
        )

        self._reranker = CrossEncoder(model_name)

    # =====================================================
    # Remove Duplicate Results
    # =====================================================

    @staticmethod
    def deduplicate(
        results: List[Dict],
    ) -> List[Dict]:
        """
        Remove duplicate retrieved chunks.
        """

        unique = OrderedDict()

        for item in results:

            key = (
                item["content"],
                item["metadata"].get(
                    "document_id",
                ),
            )

            if key not in unique:

                unique[key] = item

        return list(unique.values())

    # =====================================================
    # Filter By Score
    # =====================================================

    @staticmethod
    def filter_results(
        results: List[Dict],
        minimum_score: float = 0.30,
    ) -> List[Dict]:
        """
        Remove weak retrievals.
        """

        filtered = []

        for item in results:

            if item["score"] >= minimum_score:

                filtered.append(item)

        return filtered

    # =====================================================
    # Cross Encoder Reranking
    # =====================================================

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Improve ranking using CrossEncoder.
        """

        if not results:
            return []

        self.load_reranker()

        pairs = [

            (

                query,

                item["content"],

            )

            for item in results

        ]

        scores = self._reranker.predict(pairs)

        for item, score in zip(

            results,

            scores,

        ):

            item["rerank_score"] = float(score)

        results.sort(

            key=lambda x:

            x["rerank_score"],

            reverse=True,

        )

        return results[:top_k]

    # =====================================================
    # Hybrid Search
    # =====================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Pipeline:
        Semantic
            ↓
        Filter
            ↓
        Deduplicate
            ↓
        Rerank
        """

        results = self.semantic_search(

            query,

            top_k * 4,

        )

        results = self.filter_results(

            results

        )

        results = self.deduplicate(

            results

        )

        results = self.rerank(

            query,

            results,

            top_k,

        )

        return results

    # =====================================================
    # Context Sources
    # =====================================================

    @staticmethod
    def context_sources(
        results: List[Dict],
    ) -> List[Dict]:
        """
        Extract unique source metadata.
        """

        sources = []

        seen = set()

        for item in results:

            metadata = item["metadata"]

            filename = metadata.get(

                "filename",

                "Unknown",

            )

            page = metadata.get(

                "page",

                None,

            )

            key = (

                filename,

                page,

            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(

                {

                    "filename": filename,

                    "page": page,

                }

            )

        return sources

    # =====================================================
    # Build Prompt Context
    # =====================================================

    def prompt_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Retrieve context ready for PromptService.
        """

        results = self.hybrid_search(

            query,

            top_k,

        )

        return {

            "context":

                self.build_context(

                    results

                ),

            "sources":

                self.context_sources(

                    results

                ),

            "results":

                results,

        }

    # =====================================================
    # Retrieve For LLM
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Primary retrieval API used by ChatService.
        """

        logger.info(

            "Retrieving context..."

        )

        return self.prompt_context(

            query,

            top_k,

        )

    # =====================================================
    # Build Citation Text
    # =====================================================

    @staticmethod
    def citations(
        results: List[Dict],
    ) -> str:
        """
        Build readable citation block.
        """

        lines = [

            "Sources:",

        ]

        seen = set()

        for item in results:

            meta = item["metadata"]

            filename = meta.get(

                "filename",

                "Unknown",

            )

            page = meta.get(

                "page",

                None,

            )

            key = (

                filename,

                page,

            )

            if key in seen:
                continue

            seen.add(key)

            if page is None:

                lines.append(

                    f"- {filename}"

                )

            else:

                lines.append(

                    f"- {filename} (Page {page})"

                )

        return "\n".join(lines)


# =====================================================
# Delete Document
# =====================================================

def delete_document(
    self,
    document_id: str,
) -> bool:
    """
    Delete all chunks belonging to a document.
    """

    try:

        self.collection.delete(

            where={

                "document_id": document_id

            }

        )

        logger.info(
            "Deleted document %s",
            document_id,
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# =====================================================
# Delete By Chunk IDs
# =====================================================

def delete_chunks(
    self,
    chunk_ids: List[str],
):
    """
    Delete selected chunks.
    """

    self.collection.delete(

        ids=chunk_ids

    )

    logger.info(
        "%d chunks deleted.",
        len(chunk_ids),
    )


# =====================================================
# Update Document
# =====================================================

def update_document(
    self,
    document_id: str,
    processed_document: Dict,
):
    """
    Replace existing indexed document.
    """

    self.delete_document(document_id)

    self.index_document(processed_document)

    logger.info(
        "Document updated."
    )


# =====================================================
# Collection Statistics
# =====================================================

def statistics(self) -> Dict:

    return {

        "documents":

            self.collection.count(),

        "collection":

            self.collection_name,

        "embedding_model":

            self.embedding_service.model_name_info(),

        "persist_directory":

            self.persist_directory,

    }


# =====================================================
# Diagnostics
# =====================================================

def diagnostics(self) -> Dict:

    diagnostics = {

        "health":

            self.health(),

        "statistics":

            self.statistics(),

        "embedding":

            self.embedding_service.health(),

    }

    return diagnostics


# =====================================================
# Export Collection
# =====================================================

def export_metadata(self):

    """
    Export metadata only.
    """

    results = self.collection.get(

        include=[

            "metadatas",

        ]

    )

    return results.get(

        "metadatas",

        [],

    )


# =====================================================
# List Indexed Documents
# =====================================================

def indexed_documents(self):

    """
    List indexed document IDs.
    """

    metadata = self.export_metadata()

    ids = set()

    for item in metadata:

        if not item:
            continue

        document_id = item.get(

            "document_id"

        )

        if document_id:

            ids.add(document_id)

    return sorted(ids)


# =====================================================
# Clear Collection
# =====================================================

def clear(self):

    """
    Remove all indexed vectors.
    """

    self.reset_collection()


# =====================================================
# Collection Exists
# =====================================================

def exists(self) -> bool:

    try:

        self.client.get_collection(

            self.collection_name

        )

        return True

    except Exception:

        return False


# =====================================================
# Collection Size
# =====================================================

def size(self) -> int:

    return self.collection.count()


# =====================================================
# Service Version
# =====================================================

@staticmethod
def version():

    return "1.0.0"


# =====================================================
# Service Information
# =====================================================

def info(self):

    return {

        "service":

            "RagService",

        "version":

            self.version(),

        "vector_database":

            "ChromaDB",

        "collection":

            self.collection_name,

        "embedding":

            self.embedding_service.model_name_info(),

        "documents":

            self.collection.count(),

    }


# =====================================================
# Shutdown
# =====================================================

def close(self):

    """
    Shutdown service.
    """

    logger.info(

        "Closing RAG service."

    )

    if hasattr(

        self.embedding_service,

        "close",

    ):

        self.embedding_service.close()
