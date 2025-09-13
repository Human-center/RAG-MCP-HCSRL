# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **Phase 1 RAG Database and Model Context Protocol (MCP) Server** - a simple setup designed for ingesting, understanding, and querying unstructured text data using local storage. The initial focus is on PDFs and research papers with a straightforward Chroma small local embedding + ChromaDB + MCP architecture.

## Phase 1 Architecture

This first phase implements a minimal viable RAG system with three core components working together:

### 1. Simple Ingestion Pipeline
- **PDF Text Extraction**: Direct text extraction from PDF files using PyPDF2/PyMuPDF
- **Text to Embeddings**: Convert extracted text directly to embeddings using Chroma’s built-in small SentenceTransformer model (`all-MiniLM-L6-v2`)
- **Local Storage**: Store embeddings with basic PDF metadata in local ChromaDB instance

### 2. Local ChromaDB Storage
- **File-based Storage**: All vector data stored locally on filesystem
- **Simple Schema**: PDF text embeddings with metadata (filename, page_number)
- **No External Dependencies**: Completely self-contained local database
- **Easy Setup**: Single-command initialization with persist_directory

### 3. MCP Server Interface
- **RAG Database Interface**: Exposes ChromaDB through MCP protocol tools
- **Query Processing**: Converts natural language queries to vector searches
- **Context Retrieval**: Returns relevant document chunks as context
- **Claude Integration**: Direct integration with Claude Desktop via MCP protocol

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -U sentence-transformers
pip install torch torchvision torchaudio
pip install httpx numpy pandas
pip install chromadb python-dotenv
pip install fastapi uvicorn "mcp[cli]"
pip install pypdf2 pymupdf python-multipart
```

## Key Implementation Details

### Chroma Embedding (Small, Local)
- Model: `all-MiniLM-L6-v2` via `chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`
- Dimension: 384 (small and fast; good default for local dev)
- No external API keys; runs locally with `sentence-transformers`
- Suitable for retrieval (queries/documents) and general similarity tasks

### MCP Server Implementation
- Uses FastMCP framework with Python type hints and docstrings for automatic tool definition
- Implements tools for weather data (example from mcpserver.md)
- STDIO transport for communication with Claude Desktop and other MCP clients
- Proper error handling and logging to stderr (never stdout for STDIO servers)

### Local ChromaDB Operations
- Simple local ChromaDB instance with persistent file storage
- Basic similarity search using cosine similarity
- Minimal metadata schema (filename, page_number)
- File-based persistence in `./chroma_db/` directory

## Phase 1 Development Workflow

```bash
# Activate environment
source venv/bin/activate

# Initialize local ChromaDB (creates ./chroma_db/ directory)
python init_chroma.py

# Ingest PDFs into local ChromaDB (extracts text and creates embeddings)
python ingest_pdfs.py --input-dir ./documents

# Run MCP server to interface with RAG database
python rag_mcp_server.py

# Test complete RAG pipeline
python test_rag_query.py --query "protein folding simulations"
```

## Claude Desktop Integration

For testing MCP server with Claude Desktop, configure `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["-m", "mcp.server.fastmcp", "rag_server.py"],
      "cwd": "/absolute/path/to/RAG-MCP-HCSRL"
    }
  }
}
```

## Phase 1 Research Paper Analysis

This first phase establishes a simple local RAG system for research paper analysis:
- **Local Document Storage**: PDF research papers stored and indexed locally
- **Basic Semantic Search**: Simple queries like "protein folding methods" or "AlphaFold applications"
- **Direct Claude Integration**: MCP server provides Claude Desktop access to local document knowledge
- **Foundation for Scaling**: Simple architecture that can be extended in future phases

## Phase 1 Limitations & Future Phases

**Current Phase 1 Scope:**
- Local file storage only
- Single ChromaDB instance
- Simple PDF text extraction and embedding
- Simple MCP tool interface

**Future Phases Will Add:**
- Distributed vector storage
- Advanced chunking strategies  
- Multi-document synthesis
- Web-based UI
- Production scaling capabilities

## Note on Gemma Files

All EmbeddingGemma-specific code and docs have been moved under `./gemmaembed/` to keep the core pipeline focused on Chroma’s built-in small embedding model for local development.
