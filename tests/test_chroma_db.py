#!/usr/bin/env python3
"""
Test script for ChromaDB functionality

This script demonstrates and tests various ChromaDB operations:
- Database initialization
- Document addition with metadata
- Similarity search with and without filters
- Collection statistics
- Document deletion
- Error handling

Usage:
    python test_chroma_db.py [--reset]
"""

import argparse
import logging
import sys
import random
from typing import List, Dict, Any

from chroma_db import ChromaDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def generate_test_embeddings(count: int, dim: int = 384) -> List[List[float]]:
    """Generate random embeddings for testing."""
    return [[random.random() for _ in range(dim)] for _ in range(count)]


def test_basic_operations(db_manager: ChromaDBManager) -> bool:
    """Test basic ChromaDB operations."""
    logger.info("=== Testing Basic Operations ===")
    
    try:
        # Test initialization
        db_manager.initialize_db()
        initial_stats = db_manager.get_collection_stats()
        logger.info(f"Initial document count: {initial_stats['count']}")
        
        # Prepare test data
        test_docs = [
            "Protein structure prediction using neural networks",
            "CRISPR-Cas9 gene editing applications in medicine",
            "Quantum computing algorithms for optimization",
            "Deep learning for image recognition in medical diagnosis"
        ]
        
        test_metadata = [
            {"filename": "neural_proteins.pdf", "page_number": 1, "author": "Smith et al."},
            {"filename": "crispr_medicine.pdf", "page_number": 2, "author": "Johnson et al."},
            {"filename": "quantum_opt.pdf", "page_number": 3, "author": "Chen et al."},
            {"filename": "medical_ai.pdf", "page_number": 1, "author": "Wilson et al."}
        ]
        
        embeddings = generate_test_embeddings(len(test_docs))
        
        # Test document addition
        logger.info("Adding test documents...")
        success = db_manager.add_documents(
            embeddings=embeddings,
            documents=test_docs,
            metadatas=test_metadata
        )
        
        if not success:
            logger.error("Failed to add documents")
            return False
        
        # Verify addition
        stats = db_manager.get_collection_stats()
        expected_count = initial_stats['count'] + len(test_docs)
        if stats['count'] != expected_count:
            logger.error(f"Expected {expected_count} documents, got {stats['count']}")
            return False
        
        logger.info("Basic operations test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Basic operations test failed: {e}")
        return False


def test_similarity_search(db_manager: ChromaDBManager) -> bool:
    """Test similarity search functionality."""
    logger.info("=== Testing Similarity Search ===")
    
    try:
        # Generate query embedding
        query_embedding = generate_test_embeddings(1, 384)[0]
        
        # Test basic similarity search
        results = db_manager.similarity_search(
            query_embedding=query_embedding,
            n_results=3
        )
        
        if not results['ids']:
            logger.error("Similarity search returned no results")
            return False
        
        logger.info(f"Basic search returned {len(results['ids'])} results")
        
        # Test filtered search
        filter_results = db_manager.similarity_search(
            query_embedding=query_embedding,
            n_results=5,
            where_filter={"page_number": 1}
        )
        
        logger.info(f"Filtered search returned {len(filter_results['ids'])} results")
        
        # Verify filter worked
        for metadata in filter_results['metadatas']:
            if metadata.get('page_number') != 1:
                logger.error(f"Filter failed: found page_number={metadata.get('page_number')}")
                return False
        
        logger.info("Similarity search test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Similarity search test failed: {e}")
        return False


def test_error_handling(db_manager: ChromaDBManager) -> bool:
    """Test error handling and validation."""
    logger.info("=== Testing Error Handling ===")
    
    try:
        # Test mismatched input lengths
        try:
            db_manager.add_documents(
                embeddings=[[1, 2, 3]],
                documents=["doc1", "doc2"],  # Length mismatch
                metadatas=[{"filename": "test.pdf", "page_number": 1}]
            )
            logger.error("Should have raised ValueError for mismatched lengths")
            return False
        except ValueError:
            logger.info("Correctly caught mismatched input lengths")
        
        # Test missing required metadata
        try:
            db_manager.add_documents(
                embeddings=[[1, 2, 3]],
                documents=["test doc"],
                metadatas=[{"missing_required_fields": True}]  # Missing filename, page_number
            )
            logger.error("Should have raised ValueError for missing metadata")
            return False
        except ValueError:
            logger.info("Correctly caught missing required metadata")
        
        logger.info("Error handling test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


def test_document_management(db_manager: ChromaDBManager) -> bool:
    """Test document deletion and collection management."""
    logger.info("=== Testing Document Management ===")
    
    try:
        # Get current stats
        stats_before = db_manager.get_collection_stats()
        logger.info(f"Documents before deletion: {stats_before['count']}")
        
        # Get some document IDs for deletion
        if stats_before['count'] > 0:
            # Perform a search to get some IDs
            query_embedding = generate_test_embeddings(1, 384)[0]
            results = db_manager.similarity_search(
                query_embedding=query_embedding,
                n_results=2
            )
            
            if results['ids']:
                ids_to_delete = results['ids'][:1]  # Delete just one document
                logger.info(f"Deleting document: {ids_to_delete[0]}")
                
                success = db_manager.delete_documents(ids_to_delete)
                if not success:
                    logger.error("Failed to delete document")
                    return False
                
                # Verify deletion
                stats_after = db_manager.get_collection_stats()
                if stats_after['count'] != stats_before['count'] - 1:
                    logger.error(f"Deletion verification failed: {stats_after['count']} != {stats_before['count'] - 1}")
                    return False
                
                logger.info("Document deletion successful")
        
        logger.info("Document management test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Document management test failed: {e}")
        return False


def main():
    """Run comprehensive tests for ChromaDB functionality."""
    parser = argparse.ArgumentParser(description="Test ChromaDB functionality")
    parser.add_argument("--reset", action="store_true", help="Reset database before testing")
    parser.add_argument("--collection-name", default="test_collection", help="Test collection name")
    args = parser.parse_args()
    
    try:
        logger.info("=== ChromaDB Comprehensive Test Suite ===")
        
        # Initialize database manager
        db_manager = ChromaDBManager(collection_name=args.collection_name)
        db_manager.initialize_db()
        
        # Reset if requested
        if args.reset:
            logger.info("Resetting test collection...")
            db_manager.reset_collection()
        
        # Run tests
        tests = [
            test_basic_operations,
            test_similarity_search,
            test_error_handling,
            test_document_management
        ]
        
        results = []
        for test_func in tests:
            result = test_func(db_manager)
            results.append(result)
            if not result:
                logger.error(f"Test {test_func.__name__} failed!")
        
        # Clean up
        db_manager.close()
        
        # Report results
        passed = sum(results)
        total = len(results)
        logger.info(f"=== Test Results: {passed}/{total} tests passed ===")
        
        if passed == total:
            logger.info("All tests passed successfully!")
            return 0
        else:
            logger.error(f"{total - passed} tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
