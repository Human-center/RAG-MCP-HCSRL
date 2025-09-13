#!/usr/bin/env python3
"""
Test RAG Query Pipeline

This script tests the complete RAG pipeline by querying the ChromaDB collection
and retrieving relevant document chunks for a given query.

Usage:
    python test_rag_query.py --query "protein folding simulations"
    python test_rag_query.py --query "machine learning" --top-k 10
    python test_rag_query.py --query "AlphaFold" --chroma-dir ./custom_chroma_db
"""

import argparse
import logging
import sys
import time
from typing import List, Dict, Tuple
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    import numpy as np
except ImportError as e:
    print(f"Error: Required libraries not installed. Please install: {e}", file=sys.stderr)
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


class RAGQuerySystem:
    """RAG Query System for testing document retrieval."""

    def __init__(self, 
                 chroma_persist_dir: str = "./chroma_db",
                 collection_name: str = "pdf_documents"):
        """
        Initialize RAG query system.
        
        Args:
            chroma_persist_dir: Directory for ChromaDB persistence
            collection_name: Name of ChromaDB collection
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize ChromaDB connection."""
        if not Path(self.chroma_persist_dir).exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {self.chroma_persist_dir}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_collection(name=self.collection_name)
            collection_count = self.collection.count()
            
            self.logger.info(f"Connected to ChromaDB collection '{self.collection_name}'")
            self.logger.info(f"Collection contains {collection_count} documents")
            
            if collection_count == 0:
                self.logger.warning("Collection is empty - no documents to query")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def query(self, 
              query_text: str, 
              top_k: int = 5,
              include_metadata: bool = True) -> List[Dict]:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query_text: The query string
            top_k: Number of top results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of dictionaries containing results
        """
        self.logger.info(f"Processing query: '{query_text}'")
        
        # Search ChromaDB (collection embeds query with its embedding function)
        start_time = time.time()
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            search_time = time.time() - start_time
            
            self.logger.debug(f"Vector search completed in {search_time:.3f} seconds")
            
        except Exception as e:
            self.logger.error(f"ChromaDB query failed: {e}")
            raise
        
        # Process results
        processed_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                result = {
                    'rank': i + 1,
                    'document': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                
                if include_metadata and results['metadatas'][0][i]:
                    result['metadata'] = results['metadatas'][0][i]
                
                processed_results.append(result)
        
        self.logger.info(f"Retrieved {len(processed_results)} results")
        return processed_results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            peek = self.collection.peek(limit=5)
            
            # Calculate some basic stats
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.chroma_persist_dir
            }
            
            if peek.get('metadatas') and len(peek['metadatas']) > 0:
                # Get unique filenames
                filenames = set()
                page_numbers = []
                
                for metadata in peek['metadatas']:
                    if 'filename' in metadata:
                        filenames.add(metadata['filename'])
                    if 'page_number' in metadata:
                        page_numbers.append(metadata['page_number'])
                
                stats['sample_filenames'] = list(filenames)
                stats['sample_metadata_keys'] = list(peek['metadatas'][0].keys())
                
                # Get all documents to calculate comprehensive stats
                if count <= 1000:  # Only for small collections
                    all_results = self.collection.get()
                    if all_results.get('metadatas'):
                        all_filenames = set()
                        all_pages = set()
                        
                        for metadata in all_results['metadatas']:
                            if 'filename' in metadata:
                                all_filenames.add(metadata['filename'])
                            if 'page_number' in metadata:
                                all_pages.add(metadata['page_number'])
                        
                        stats['unique_files'] = len(all_filenames)
                        stats['unique_pages'] = len(all_pages)
                        stats['all_filenames'] = sorted(list(all_filenames))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}


def main():
    """Main entry point for RAG query testing."""
    parser = argparse.ArgumentParser(
        description="Test RAG Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Query for protein folding information
    python test_rag_query.py --query "protein folding simulations"
    
    # Get more results
    python test_rag_query.py --query "machine learning" --top-k 10
    
    # Use custom ChromaDB location
    python test_rag_query.py --query "AlphaFold" --chroma-dir ./custom_chroma_db
    
    # Show collection statistics
    python test_rag_query.py --show-stats
    
    # Run interactive mode
    python test_rag_query.py --interactive
        """
    )
    
    # Query options
    parser.add_argument(
        "--query",
        help="Query string to search for"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    
    # Storage options
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
    
    # (No embedding model option; uses MiniLM via Chroma)
    
    # Utility options
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show collection statistics and exit"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize RAG system
        rag_system = RAGQuerySystem(
            chroma_persist_dir=args.chroma_dir,
            collection_name=args.collection_name
        )
        
        # Handle statistics request
        if args.show_stats:
            stats = rag_system.get_collection_stats()
            print("\n=== Collection Statistics ===")
            for key, value in stats.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"{key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"{key}: {value}")
            return
        
        # Handle interactive mode
        if args.interactive:
            print("=== Interactive RAG Query Mode ===")
            print("Enter queries to search the document collection.")
            print("Type 'quit', 'exit', or press Ctrl+C to exit.")
            print()
            
            while True:
                try:
                    query = input("Query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    results = rag_system.query(query, top_k=args.top_k)
                    
                    if not results:
                        print("No results found.\n")
                        continue
                    
                    print(f"\nFound {len(results)} results:")
                    print("-" * 80)
                    
                    for result in results:
                        print(f"Rank {result['rank']} | Similarity: {result['similarity']:.4f}")
                        
                        if 'metadata' in result:
                            metadata = result['metadata']
                            print(f"Source: {metadata.get('filename', 'Unknown')} (Page {metadata.get('page_number', 'Unknown')})")
                        
                        # Truncate long documents
                        doc_text = result['document']
                        if len(doc_text) > 300:
                            doc_text = doc_text[:300] + "..."
                        
                        print(f"Text: {doc_text}")
                        print("-" * 40)
                    
                    print()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    continue
            
            print("Goodbye!")
            return
        
        # Handle single query
        if not args.query:
            parser.error("Either --query, --show-stats, or --interactive must be specified")
        
        # Process single query
        logger.info("Processing single query")
        results = rag_system.query(args.query, top_k=args.top_k)
        
        # Display results
        print(f"\n=== Query Results for: '{args.query}' ===")
        
        if not results:
            print("No results found.")
            return
        
        print(f"Found {len(results)} results:\n")
        
        for result in results:
            print(f"Rank {result['rank']} | Similarity: {result['similarity']:.4f} | Distance: {result['distance']:.4f}")
            
            if 'metadata' in result:
                metadata = result['metadata']
                filename = metadata.get('filename', 'Unknown')
                page_num = metadata.get('page_number', 'Unknown')
                chunk_idx = metadata.get('chunk_index', 'Unknown')
                char_len = metadata.get('char_length', 'Unknown')
                
                print(f"Source: {filename} (Page {page_num}, Chunk {chunk_idx}, {char_len} chars)")
            
            # Show document text (truncated if too long)
            doc_text = result['document']
            if len(doc_text) > 500:
                doc_text = doc_text[:500] + "..."
            
            print(f"Text: {doc_text}")
            print("-" * 80)
        
        # Summary statistics
        similarities = [r['similarity'] for r in results]
        print(f"\nSummary Statistics:")
        print(f"Average similarity: {np.mean(similarities):.4f}")
        print(f"Max similarity: {np.max(similarities):.4f}")
        print(f"Min similarity: {np.min(similarities):.4f}")
        
        # Show unique sources
        if results and 'metadata' in results[0]:
            sources = set()
            for result in results:
                if 'metadata' in result and 'filename' in result['metadata']:
                    sources.add(result['metadata']['filename'])
            print(f"Unique sources: {len(sources)}")
            if len(sources) <= 5:
                for source in sorted(sources):
                    print(f"  - {source}")
        
    except KeyboardInterrupt:
        logger.info("Query testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Query testing failed: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
