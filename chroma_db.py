"""
ChromaDB Local Database Setup for RAG System

This module implements a local ChromaDB instance for storing and retrieving PDF document embeddings
in the Phase 1 RAG system. It provides functionality for:
- Setting up local ChromaDB with persistent file storage
- Creating collections for PDF document storage
- Storing embeddings with metadata (filename, page_number, content)
- Similarity search using cosine similarity
- Database initialization and connection management

Requirements:
- chromadb: pip install chromadb
- sentence-transformers: pip install sentence-transformers
"""

import os
import logging
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError(
        "ChromaDB is not installed. Install it with: pip install chromadb"
    )

# Configure logging to stderr (important for MCP servers)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Manages local ChromaDB instance for RAG document storage and retrieval.
    
    This class provides a simple interface for storing PDF document embeddings
    with metadata and performing similarity searches for RAG applications.
    """
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db/",
        collection_name: str = "pdf_documents"
    ):
        """
        Initialize ChromaDB manager with local persistent storage.
        
        Args:
            persist_directory: Directory path for ChromaDB persistence (default: "./chroma_db/")
            collection_name: Name of the collection for storing documents (default: "pdf_documents")
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChromaDB persist directory: {self.persist_directory.absolute()}")
        
    def initialize_db(self) -> None:
        """
        Initialize ChromaDB client and create/get collection.
        
        Sets up the local ChromaDB instance with persistent storage and creates
        the main collection for storing PDF document embeddings.
        
        Raises:
            Exception: If database initialization fails
        """
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory.absolute()),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Use SentenceTransformer embedding function for consistent 384-dim embeddings
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get collection for PDF documents
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG system"},
                embedding_function=embedding_function
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            logger.info(f"Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise Exception(f"ChromaDB initialization failed: {e}")
    
    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents with embeddings and metadata to the collection.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document text content
            metadatas: List of metadata dictionaries (must include filename, page_number)
            ids: Optional list of unique IDs for documents (auto-generated if None)
        
        Returns:
            bool: True if documents were added successfully, False otherwise
        
        Raises:
            ValueError: If input lists have different lengths or required metadata is missing
            Exception: If database operation fails
        """
        if not self.collection:
            raise Exception("Database not initialized. Call initialize_db() first.")
        
        # Validate input lengths
        if not (len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("Embeddings, documents, and metadatas must have the same length")
        
        if ids and len(ids) != len(documents):
            raise ValueError("IDs list must have the same length as other inputs")
        
        # Validate required metadata fields
        required_fields = ["filename", "page_number"]
        for i, metadata in enumerate(metadatas):
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required metadata field '{field}' in document {i}")
        
        try:
            # Generate IDs if not provided
            if not ids:
                ids = [f"{meta['filename']}_page_{meta['page_number']}_{i}" 
                       for i, meta in enumerate(metadatas)]
            
            # Add documents to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def similarity_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform similarity search using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (default: 5)
            where_filter: Optional metadata filter (e.g., {"filename": "document.pdf"})
        
        Returns:
            Dict containing:
                - ids: List of document IDs
                - documents: List of document texts
                - metadatas: List of metadata dictionaries
                - distances: List of cosine distances (lower = more similar)
        
        Raises:
            Exception: If database operation fails
        """
        if not self.collection:
            raise Exception("Database not initialized. Call initialize_db() first.")
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results for easier use
            formatted_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
            logger.info(f"Similarity search returned {len(formatted_results['ids'])} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise Exception(f"Similarity search failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict containing collection statistics:
                - count: Total number of documents
                - name: Collection name
                - metadata: Collection metadata
        """
        if not self.collection:
            raise Exception("Database not initialized. Call initialize_db() first.")
        
        try:
            stats = {
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise Exception(f"Failed to get collection stats: {e}")
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self.collection:
            raise Exception("Database not initialized. Call initialize_db() first.")
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset (clear) the entire collection.
        
        WARNING: This will delete all documents in the collection.
        
        Returns:
            bool: True if reset was successful, False otherwise
        """
        if not self.collection:
            raise Exception("Database not initialized. Call initialize_db() first.")
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG system"}
            )
            
            logger.info(f"Collection {self.collection_name} has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def close(self) -> None:
        """
        Close the database connection and clean up resources.
        """
        if self.client:
            # ChromaDB client doesn't require explicit closing
            self.client = None
            self.collection = None
            logger.info("ChromaDB connection closed")


def create_sample_documents() -> Tuple[List[List[float]], List[str], List[Dict[str, Any]]]:
    """
    Create sample documents for testing purposes.
    
    Returns:
        Tuple of (embeddings, documents, metadatas) for testing
    """
    import random
    
    # Sample documents
    documents = [
        "Protein folding is a fundamental biological process that determines protein structure.",
        "AlphaFold has revolutionized protein structure prediction using deep learning methods.",
        "Molecular dynamics simulations provide insights into protein behavior over time.",
        "CRISPR gene editing technology allows precise modification of DNA sequences.",
        "Machine learning algorithms are increasingly used in drug discovery pipelines."
    ]
    
    # Generate random embeddings (in real use, embeddings are computed via Chroma)
    embedding_dim = 384  # MiniLM dimension
    embeddings = [[random.random() for _ in range(embedding_dim)] for _ in documents]
    
    # Sample metadata
    metadatas = [
        {"filename": "protein_folding.pdf", "page_number": 1, "topic": "biology"},
        {"filename": "alphafold_paper.pdf", "page_number": 3, "topic": "ai"},
        {"filename": "md_simulations.pdf", "page_number": 2, "topic": "computational"},
        {"filename": "crispr_review.pdf", "page_number": 1, "topic": "biotechnology"},
        {"filename": "ml_drug_discovery.pdf", "page_number": 5, "topic": "ai"}
    ]
    
    return embeddings, documents, metadatas


def main():
    """
    Example usage of ChromaDB manager.
    
    Demonstrates basic operations:
    1. Initialize database
    2. Add sample documents
    3. Perform similarity search
    4. Get collection statistics
    """
    try:
        # Initialize ChromaDB manager
        logger.info("=== ChromaDB Manager Example ===")
        db_manager = ChromaDBManager()
        
        # Initialize the database
        logger.info("Initializing ChromaDB...")
        db_manager.initialize_db()
        
        # Get initial stats
        initial_stats = db_manager.get_collection_stats()
        logger.info(f"Initial collection stats: {initial_stats}")
        
        # Create and add sample documents if collection is empty
        if initial_stats["count"] == 0:
            logger.info("Adding sample documents...")
            embeddings, documents, metadatas = create_sample_documents()
            
            success = db_manager.add_documents(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            if success:
                logger.info("Sample documents added successfully!")
            else:
                logger.error("Failed to add sample documents")
                return
        
        # Get updated stats
        stats = db_manager.get_collection_stats()
        logger.info(f"Collection stats after adding documents: {stats}")
        
        # Perform a sample similarity search
        logger.info("Performing similarity search...")
        # Use the first embedding as query (in real use, this would be from a query)
        sample_embeddings, _, _ = create_sample_documents()
        query_embedding = sample_embeddings[0]
        
        results = db_manager.similarity_search(
            query_embedding=query_embedding,
            n_results=3
        )
        
        logger.info(f"Search results:")
        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(results["ids"], results["documents"], results["metadatas"], results["distances"])
        ):
            logger.info(f"  Result {i+1}:")
            logger.info(f"    ID: {doc_id}")
            logger.info(f"    Document: {document[:100]}...")
            logger.info(f"    Metadata: {metadata}")
            logger.info(f"    Distance: {distance:.4f}")
        
        # Test filtered search
        logger.info("Performing filtered search (topic='ai')...")
        filtered_results = db_manager.similarity_search(
            query_embedding=query_embedding,
            n_results=2,
            where_filter={"topic": "ai"}
        )
        
        logger.info(f"Filtered search returned {len(filtered_results['ids'])} results")
        
        # Close the database connection
        db_manager.close()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
