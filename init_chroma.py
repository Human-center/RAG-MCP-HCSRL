#!/usr/bin/env python3
"""
ChromaDB Initialization Script

This script initializes a local ChromaDB instance for the RAG system.
It creates the necessary directory structure and collection for storing PDF document embeddings.

Usage:
    python init_chroma.py
    python init_chroma.py --chroma-dir ./custom_chroma_db
    python init_chroma.py --collection-name custom_collection
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Error: ChromaDB not installed. Please install: pip install chromadb", file=sys.stderr)
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def initialize_chromadb(persist_directory: str, collection_name: str, reset: bool = False):
    """
    Initialize ChromaDB with the specified configuration.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection to create
        reset: Whether to reset (delete) existing data
    """
    logger = logging.getLogger(__name__)
    
    # Ensure persist directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using persist directory: {persist_directory}")
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Reset if requested
        if reset:
            logger.warning("Resetting ChromaDB - all existing data will be deleted!")
            confirmation = input("Are you sure you want to reset the database? (y/N): ")
            if confirmation.lower() in ['y', 'yes']:
                client.reset()
                logger.info("ChromaDB reset completed")
            else:
                logger.info("Reset cancelled")
                return
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Found existing collection '{collection_name}' with {collection.count()} documents")
        except ValueError:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "description": "PDF document chunks for RAG system",
                    "created_by": "init_chroma.py"
                }
            )
            logger.info(f"Created new collection '{collection_name}'")
        
        # Display collection information
        print("\n=== ChromaDB Initialization Complete ===")
        print(f"Persist Directory: {persist_directory}")
        print(f"Collection Name: {collection_name}")
        print(f"Collection Count: {collection.count()}")
        
        # Show sample data if available
        if collection.count() > 0:
            peek = collection.peek(limit=1)
            if peek.get("metadatas") and len(peek["metadatas"]) > 0:
                print(f"Sample metadata keys: {list(peek['metadatas'][0].keys())}")
        
        logger.info("ChromaDB initialization successful")
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


def main():
    """Main entry point for ChromaDB initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize ChromaDB for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize with default settings
    python init_chroma.py
    
    # Use custom directory
    python init_chroma.py --chroma-dir ./my_chroma_db
    
    # Use custom collection name
    python init_chroma.py --collection-name research_papers
    
    # Reset existing database
    python init_chroma.py --reset
        """
    )
    
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db",
        help="Directory for ChromaDB persistence (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection-name",
        default="pdf_documents",
        help="Name of ChromaDB collection (default: pdf_documents)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset (delete) existing ChromaDB data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        initialize_chromadb(
            persist_directory=args.chroma_dir,
            collection_name=args.collection_name,
            reset=args.reset
        )
        
    except KeyboardInterrupt:
        print("\nInitialization interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()