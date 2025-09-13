# RAG System PDF Ingestion Pipeline - Setup Guide

This guide helps you set up and use the PDF ingestion pipeline for the RAG (Retrieval-Augmented Generation) system.

## Quick Start

1. **Setup Environment**:
   ```bash
   # Activate your virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Run setup script
   python setup.py
   ```

2. **Authenticate with Hugging Face** (required for EmbeddingGemma):
   ```bash
   # Visit https://huggingface.co/google/embeddinggemma-300M and accept license
   # Generate token at https://huggingface.co/settings/tokens
   huggingface-cli login
   ```

3. **Initialize ChromaDB**:
   ```bash
   python init_chroma.py
   ```

4. **Ingest PDF Documents**:
   ```bash
   # Place your PDF files in the ./documents directory
   python ingest_pdfs.py --input-dir ./documents
   ```

5. **Test the System**:
   ```bash
   python test_rag_query.py --query "protein folding simulations"
   ```

## Detailed Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- GPU with CUDA support (optional, for faster processing)

### Installation Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python setup.py --check-only
   ```

3. **Test Components**:
   ```bash
   # Test embedding model
   python test_embeddings.py
   
   # Test ChromaDB initialization
   python init_chroma.py --verbose
   ```

## Usage Examples

### PDF Ingestion

```bash
# Process a single PDF
python ingest_pdfs.py --input-file document.pdf

# Process all PDFs in a directory
python ingest_pdfs.py --input-dir ./documents

# Custom chunk size and overlap
python ingest_pdfs.py --input-dir ./documents --chunk-size 1500 --overlap 300

# Use custom ChromaDB location
python ingest_pdfs.py --input-dir ./documents --chroma-dir ./my_chroma_db

# Verbose output with progress tracking
python ingest_pdfs.py --input-dir ./documents --verbose
```

### Querying the System

```bash
# Simple query
python test_rag_query.py --query "machine learning applications"

# Get more results
python test_rag_query.py --query "protein folding" --top-k 10

# Interactive mode
python test_rag_query.py --interactive

# Show collection statistics
python test_rag_query.py --show-stats
```

### ChromaDB Management

```bash
# Initialize with default settings
python init_chroma.py

# Use custom directory and collection name
python init_chroma.py --chroma-dir ./custom_db --collection-name research_papers

# Reset existing database
python init_chroma.py --reset

# View current collection info
python ingest_pdfs.py --info
```

## System Architecture

The ingestion pipeline consists of several key components:

### 1. **PDF Text Extraction**
- Uses PyPDF2 and PyMuPDF with fallback strategy
- Handles corrupted or difficult-to-read PDFs
- Extracts text page by page with metadata

### 2. **Intelligent Text Chunking**
- Semantic chunking based on sentence boundaries
- Configurable chunk size and overlap
- Preserves document structure and context

### 3. **Embedding Generation**
- Uses EmbeddingGemma model (google/embeddinggemma-300M)
- Task-specific prompts for retrieval optimization
- Batch processing for efficiency

### 4. **Vector Storage**
- Local ChromaDB instance with file persistence
- Rich metadata storage (filename, page_number, chunk_id)
- Efficient similarity search capabilities

## Configuration Options

### Ingestion Parameters
- `--chunk-size`: Target size of text chunks (default: 1000 characters)
- `--overlap`: Overlap between chunks (default: 200 characters)
- `--max-workers`: Parallel processing threads (default: 4)

### Storage Options
- `--chroma-dir`: ChromaDB storage directory (default: ./chroma_db)
- `--collection-name`: Collection name (default: pdf_documents)

### Model Options
- `--embedding-model`: EmbeddingGemma model ID (default: google/embeddinggemma-300M)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   python setup.py --install-deps
   ```

2. **Hugging Face Authentication**:
   ```bash
   huggingface-cli login
   # Visit https://huggingface.co/google/embeddinggemma-300M first
   ```

3. **GPU Not Detected**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **PDF Processing Errors**:
   - Check PDF file integrity
   - Try different PDF files
   - Use `--verbose` flag for detailed error messages

5. **Memory Issues**:
   - Reduce `--chunk-size` and `--max-workers`
   - Process PDFs in smaller batches
   - Use CPU instead of GPU for large documents

### Performance Tips

1. **GPU Usage**: System automatically uses GPU if available
2. **Batch Size**: Adjust embedding batch size in code for memory optimization
3. **Parallel Processing**: Tune `--max-workers` based on your system
4. **Chunk Size**: Larger chunks = fewer chunks but potentially less precise retrieval

## File Structure

```
RAG-MCP-HCSRL/
├── ingest_pdfs.py          # Main ingestion pipeline
├── init_chroma.py          # ChromaDB initialization
├── test_embeddings.py      # Embedding model testing
├── test_rag_query.py       # RAG query testing
├── setup.py               # Setup and verification script
├── requirements.txt       # Python dependencies
├── README_SETUP.md        # This file
├── chroma_db/            # ChromaDB storage (created automatically)
└── documents/            # Place your PDF files here
```

## Advanced Usage

### Custom Chunking Strategy
Modify the `TextChunker` class in `ingest_pdfs.py` to implement custom chunking logic.

### Different Embedding Models
Change the `--embedding-model` parameter to use different sentence transformer models.

### Integration with MCP Server
The processed documents will be available through the MCP server interface for Claude Desktop integration.

## Support

For issues or questions:
1. Check the verbose logs with `--verbose` flag
2. Verify system setup with `python setup.py --check-only`
3. Test individual components with provided test scripts