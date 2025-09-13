#!/usr/bin/env python3
"""
PDF Ingestion Pipeline for RAG System

This module implements a comprehensive PDF ingestion pipeline that:
1. Extracts text from PDF files using PyPDF2/PyMuPDF
2. Implements intelligent text chunking for semantic coherence
3. Generates embeddings using EmbeddingGemma model
4. Stores processed documents in ChromaDB with metadata
5. Handles batch processing with progress tracking and error handling

Usage:
    python ingest_pdfs.py --input-dir ./documents
    python ingest_pdfs.py --input-file document.pdf
    python ingest_pdfs.py --input-dir ./documents --chunk-size 1000 --overlap 200
"""

import argparse
import logging
import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"Error: Missing PDF processing libraries. Please install: {e}", file=sys.stderr)
    sys.exit(1)

# Vector storage
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError as e:
    print(f"Error: Missing ML/vector storage libraries. Please install: {e}", file=sys.stderr)
    sys.exit(1)

# Utilities
try:
    from tqdm import tqdm
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError as e:
    print(f"Error: Missing utility libraries. Please install: {e}", file=sys.stderr)
    sys.exit(1)


@dataclass
class DocumentChunk:
    """Represents a chunk of text extracted from a PDF document."""
    text: str
    page_number: int
    chunk_id: str
    filename: str
    file_path: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, any]


@dataclass
class IngestionStats:
    """Statistics for the ingestion process."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    processing_time: float = 0.0
    failed_file_paths: List[str] = None

    def __post_init__(self):
        if self.failed_file_paths is None:
            self.failed_file_paths = []


class PDFExtractor:
    """Handles PDF text extraction using multiple libraries for robustness."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_pypdf2(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text using PyPDF2."""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            pages.append((page_num, text))
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num} from {pdf_path} with PyPDF2: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Failed to read {pdf_path} with PyPDF2: {e}")
            raise

        return pages

    def extract_text_pymupdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text using PyMuPDF (fitz)."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():  # Only add non-empty pages
                        pages.append((page_num + 1, text))
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num + 1} from {pdf_path} with PyMuPDF: {e}")
                    continue
            doc.close()
        except Exception as e:
            self.logger.error(f"Failed to read {pdf_path} with PyMuPDF: {e}")
            raise

        return pages

    def extract_text(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF using fallback strategy.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, text)
        """
        self.logger.info(f"Extracting text from {pdf_path}")
        
        # Try PyMuPDF first (generally more robust)
        try:
            pages = self.extract_text_pymupdf(pdf_path)
            if pages:
                self.logger.debug(f"Successfully extracted {len(pages)} pages with PyMuPDF")
                return pages
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")

        # Fallback to PyPDF2
        try:
            pages = self.extract_text_pypdf2(pdf_path)
            if pages:
                self.logger.debug(f"Successfully extracted {len(pages)} pages with PyPDF2")
                return pages
        except Exception as e:
            self.logger.error(f"Both PDF extractors failed for {pdf_path}: {e}")
            raise

        # If we get here, both methods failed to extract any text
        raise ValueError(f"No text could be extracted from {pdf_path}")


class TextChunker:
    """Implements intelligent text chunking strategies."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

    def chunk_text_semantic(self, text: str, page_number: int, filename: str, file_path: str) -> List[DocumentChunk]:
        """
        Chunk text using sentence boundaries for better semantic coherence.
        
        Args:
            text: Text to chunk
            page_number: Page number the text came from
            filename: Name of the source file
            file_path: Full path to the source file
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []

        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            self.logger.warning(f"Sentence tokenization failed, falling back to simple splitting: {e}")
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunk_id = str(uuid.uuid4())
                    char_end = current_start + len(current_chunk)
                    
                    chunk = DocumentChunk(
                        text=current_chunk.strip(),
                        page_number=page_number,
                        chunk_id=chunk_id,
                        filename=filename,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        char_start=current_start,
                        char_end=char_end,
                        metadata={
                            'chunk_type': 'semantic',
                            'sentence_count': len(sent_tokenize(current_chunk.strip())) if current_chunk.strip() else 0,
                            'char_length': len(current_chunk.strip())
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Take last `overlap` characters for overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_start = char_end - len(overlap_text)
                else:
                    current_chunk = sentence
                    current_start = char_end

        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_id = str(uuid.uuid4())
            char_end = current_start + len(current_chunk)
            
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                page_number=page_number,
                chunk_id=chunk_id,
                filename=filename,
                file_path=file_path,
                chunk_index=chunk_index,
                char_start=current_start,
                char_end=char_end,
                metadata={
                    'chunk_type': 'semantic',
                    'sentence_count': len(sent_tokenize(current_chunk.strip())) if current_chunk.strip() else 0,
                    'char_length': len(current_chunk.strip())
                }
            )
            chunks.append(chunk)

        return chunks


class ChromaDBStorage:
    """Handles ChromaDB storage operations."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "pdf_documents"):
        """
        Initialize ChromaDB storage.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Use SentenceTransformer embedding function for consistent 384-dim embeddings
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PDF document chunks for RAG system"},
                embedding_function=self.embedding_function
            )
            
            self.logger.info(f"ChromaDB initialized at {persist_directory}, collection: {collection_name}")
            self.logger.info(f"Current collection size: {self.collection.count()}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to ChromaDB.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        try:
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    "filename": chunk.filename,
                    "file_path": chunk.file_path,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "char_length": len(chunk.text),
                    "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                    "sentence_count": chunk.metadata.get("sentence_count", 0)
                }
                metadatas.append(metadata)

            # Embeddings are generated by the collection's embedding_function
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Added {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise

    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            peek = self.collection.peek()
            
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory,
                "sample_metadata": peek.get("metadatas", [{}])[0] if peek.get("metadatas") else {}
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}


class PDFIngestionPipeline:
    """Main pipeline class that orchestrates the entire ingestion process."""

    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 chroma_persist_dir: str = "./chroma_db",
                 collection_name: str = "pdf_documents",
                 max_workers: int = 4):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            chroma_persist_dir: Directory for ChromaDB persistence
            collection_name: Name of ChromaDB collection
            embedding_model: EmbeddingGemma model ID
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.storage = ChromaDBStorage(persist_directory=chroma_persist_dir, collection_name=collection_name)

    def process_single_pdf(self, pdf_path: str) -> Tuple[bool, int, str]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success, chunk_count, error_message)
        """
        try:
            pdf_path = str(Path(pdf_path).resolve())
            filename = Path(pdf_path).name
            
            self.logger.info(f"Processing {filename}")
            
            # Extract text from PDF
            pages = self.pdf_extractor.extract_text(pdf_path)
            if not pages:
                return False, 0, "No text extracted from PDF"
            
            # Process each page and create chunks
            all_chunks = []
            for page_num, text in pages:
                if not text.strip():
                    continue
                    
                page_chunks = self.text_chunker.chunk_text_semantic(
                    text=text,
                    page_number=page_num,
                    filename=filename,
                    file_path=pdf_path
                )
                all_chunks.extend(page_chunks)
            
            if not all_chunks:
                return False, 0, "No chunks created from PDF"
            
            # Store in ChromaDB (embeddings computed by Chroma)
            self.storage.add_chunks(all_chunks)
            
            self.logger.info(f"Successfully processed {filename}: {len(all_chunks)} chunks")
            return True, len(all_chunks), ""
            
        except Exception as e:
            error_msg = f"Failed to process {pdf_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return False, 0, error_msg

    def process_pdf_batch(self, pdf_paths: List[str]) -> IngestionStats:
        """
        Process a batch of PDF files with parallel processing.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            IngestionStats object with processing statistics
        """
        stats = IngestionStats()
        stats.total_files = len(pdf_paths)
        start_time = time.time()
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} PDF files")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_pdf, pdf_path): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(pdf_paths), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    
                    try:
                        success, chunk_count, error_msg = future.result()
                        
                        if success:
                            stats.processed_files += 1
                            stats.total_chunks += chunk_count
                        else:
                            stats.failed_files += 1
                            stats.failed_file_paths.append(pdf_path)
                            self.logger.error(f"Failed to process {pdf_path}: {error_msg}")
                            
                    except Exception as e:
                        stats.failed_files += 1
                        stats.failed_file_paths.append(pdf_path)
                        self.logger.error(f"Exception processing {pdf_path}: {e}")
                    
                    pbar.update(1)
        
        stats.processing_time = time.time() - start_time
        
        # Log final statistics
        self.logger.info(f"Batch processing completed in {stats.processing_time:.2f} seconds")
        self.logger.info(f"Successfully processed: {stats.processed_files}/{stats.total_files} files")
        self.logger.info(f"Total chunks created: {stats.total_chunks}")
        
        if stats.failed_files > 0:
            self.logger.warning(f"Failed to process {stats.failed_files} files:")
            for failed_path in stats.failed_file_paths:
                self.logger.warning(f"  - {failed_path}")
        
        return stats

    def get_storage_info(self) -> Dict:
        """Get information about the storage backend."""
        return self.storage.get_collection_info()


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging to stderr (important for MCP compatibility)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def find_pdf_files(input_path: str) -> List[str]:
    """
    Find PDF files in the given path.
    
    Args:
        input_path: Path to file or directory
        
    Returns:
        List of PDF file paths
    """
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix.lower() == '.pdf':
            return [str(path)]
        else:
            raise ValueError(f"Input file {input_path} is not a PDF")
    
    elif path.is_dir():
        pdf_files = []
        for pdf_path in path.rglob("*.pdf"):
            if pdf_path.is_file():
                pdf_files.append(str(pdf_path))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory {input_path}")
        
        return pdf_files
    
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def main():
    """Main entry point for the PDF ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="PDF Ingestion Pipeline for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single PDF file
    python ingest_pdfs.py --input-file document.pdf
    
    # Process all PDFs in a directory
    python ingest_pdfs.py --input-dir ./documents
    
    # Custom chunk size and overlap
    python ingest_pdfs.py --input-dir ./documents --chunk-size 1500 --overlap 300
    
    # Use custom ChromaDB location
    python ingest_pdfs.py --input-dir ./documents --chroma-dir ./my_chroma_db
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        help="Directory containing PDF files to process"
    )
    input_group.add_argument(
        "--input-file",
        help="Single PDF file to process"
    )
    
    # Processing options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target size of text chunks in characters (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel worker threads (default: 4)"
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show ChromaDB collection info and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = PDFIngestionPipeline(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            chroma_persist_dir=args.chroma_dir,
            collection_name=args.collection_name,
            max_workers=args.max_workers
        )
        
        # Handle info request
        if args.info:
            info = pipeline.get_storage_info()
            print("\n=== ChromaDB Collection Info ===")
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        # Find PDF files to process
        input_path = args.input_dir or args.input_file
        pdf_files = find_pdf_files(input_path)
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process PDFs
        stats = pipeline.process_pdf_batch(pdf_files)
        
        # Print final report
        print("\n=== Ingestion Complete ===")
        print(f"Total files: {stats.total_files}")
        print(f"Successfully processed: {stats.processed_files}")
        print(f"Failed: {stats.failed_files}")
        print(f"Total chunks created: {stats.total_chunks}")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        
        if stats.failed_files > 0:
            print(f"\nFailed files ({stats.failed_files}):")
            for failed_path in stats.failed_file_paths:
                print(f"  - {Path(failed_path).name}")
        
        # Show final collection info
        info = pipeline.get_storage_info()
        print(f"\nChromaDB Collection: {info['name']}")
        print(f"Total documents: {info['count']}")
        print(f"Storage location: {info['persist_directory']}")
        
        # Exit with appropriate code
        sys.exit(0 if stats.failed_files == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
