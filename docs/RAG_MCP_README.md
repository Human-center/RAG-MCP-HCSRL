# RAG MCP Server - Implementation Guide

## Overview

This implementation provides a **Phase 1 RAG Database and Model Context Protocol (MCP) Server** that integrates Chroma's small local embedding (all-MiniLM-L6-v2), ChromaDB, and the FastMCP framework to create a semantic search system accessible through Claude Desktop.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Claude Desktop │◄──►│  RAG MCP Server  │◄──►│   ChromaDB      │
│                 │    │                  │    │  (Vector Store) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  all-MiniLM-L6   │
                       │     (local)      │
                       └──────────────────┘
```

## Files Created

### Core Server Implementation
- **`rag_mcp_server.py`** - Main MCP server with FastMCP framework
- **`init_chroma.py`** - ChromaDB initialization script
- **`ingest_pdfs.py`** - PDF ingestion and embedding pipeline
- **`test_rag_query.py`** - RAG query testing script
  

## Features

### MCP Tools Available in Claude Desktop

1. **`query_documents`** - Semantic search over document collection
   - Natural language queries
   - Configurable similarity thresholds
   - Ranked results with metadata

2. **`get_collection_info`** - Collection statistics and status
   - Document count
   - Metadata field overview
   - Database health check

3. **`list_recent_documents`** - Browse ingested documents
   - Preview document content
   - File and page metadata
   - Quick collection overview

## Setup Instructions

### 1. Environment Setup

```bash
# Ensure you're in the project directory
cd /Users/vasanth/RAG-MCP-HCSRL

# Activate virtual environment
source venv/bin/activate

# Verify dependencies are installed
pip list | grep -E "(sentence-transformers|chromadb|mcp)"
```

### 2. Initialize ChromaDB

```bash
python init_chroma.py
```
*Creates local ChromaDB instance in `./chroma_db/` directory*

### 3. Ingest PDF Documents

```bash
# Create documents directory
mkdir -p documents

# Add your PDF files to ./documents/
# Then run ingestion
python ingest_pdfs.py --input-dir ./documents
```

### 4. Test RAG Query Pipeline

```bash
python test_rag_query.py --query "machine learning algorithms"
```

## Claude Desktop Integration

### Configuration File

Create or edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
  "args": [
    "/Users/vasanth/RAG-MCP-HCSRL/rag_mcp_server.py"
  ],
      "cwd": "/Users/vasanth/RAG-MCP-HCSRL",
      "env": {
        "PYTHONPATH": "/Users/vasanth/RAG-MCP-HCSRL/venv/lib/python3.13/site-packages"
      }
    }
  }
}
```

### Alternative Configuration (using virtual environment)

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "/Users/vasanth/RAG-MCP-HCSRL/venv/bin/python",
      "args": [
        "/Users/vasanth/RAG-MCP-HCSRL/rag_mcp_server.py"
      ],
      "cwd": "/Users/vasanth/RAG-MCP-HCSRL"
    }
  }
}
```

### Steps to Connect

1. **Save Configuration**: Add the above JSON to your claude_desktop_config.json
2. **Restart Claude Desktop**: Completely quit and restart the application
3. **Verify Connection**: Look for MCP tools in the Claude Desktop interface
4. **Test Integration**: Try asking Claude to search your documents

## Usage Examples

### In Claude Desktop

Once connected, you can use these commands:

```
Search for documents about "protein folding simulations"

Get information about my document collection

List recent documents that were ingested

Find papers related to "machine learning in biology"
```

### Direct Testing

```bash
# Test specific queries
python test_rag_query.py --query "deep learning applications" --max-results 3

# Test with different similarity thresholds
python test_rag_query.py --query "neural networks" --min-similarity 0.5

# Ingest additional documents
python ingest_pdfs.py --input-dir ./new_papers/
```

## Technical Details

### Embedding Configuration
- **Model**: `all-MiniLM-L6-v2` (via Chroma embedding_function)
- **Dimension**: 384
- **Runs locally**: No external API keys

### ChromaDB Configuration
- **Storage**: Local persistent storage in `./chroma_db/`
- **Collection**: `pdf_documents` (default)
- **Distance Metric**: Cosine similarity
- **Metadata Fields**: filename, page_number, chunk_index, file_path

### FastMCP Tools
- **Error Handling**: All errors logged to stderr, never stdout
- **Transport**: STDIO for Claude Desktop integration  
- **Type Safety**: Full Python type hints and docstrings
- **Tool Decorations**: `@mcp.tool()` with automatic schema generation

## Troubleshooting

### Common Issues

1. **"Collection not found"**
   ```bash
   python init_chroma.py
   ```

2. **"No documents in collection"**
   ```bash
   python ingest_pdfs.py --input-dir ./documents
   ```

3. **"EmbeddingGemma model access denied"**
   - Visit https://huggingface.co/google/embeddinggemma-300M and accept license
   - Generate token at https://huggingface.co/settings/tokens
   - Run `huggingface-cli login`

4. **"Claude Desktop not showing MCP tools"**
   - Verify config file path: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Check absolute paths in configuration
   - Restart Claude Desktop completely
   - Check Claude Desktop logs for connection errors

### Logging

All server logs are written to stderr and can be monitored:

```bash
# Run server directly to see logs
python rag_mcp_server.py

# Or check system logs when running through Claude Desktop
tail -f /var/log/system.log | grep -i claude
```

## Performance Notes

### Optimization Tips

1. **GPU Acceleration**: EmbeddingGemma will use CUDA if available
2. **Batch Processing**: Ingestion processes documents in batches
3. **Chunking Strategy**: 1000 character chunks with 200 character overlap
4. **Memory Usage**: ~200MB RAM for EmbeddingGemma with quantization

### Scaling Considerations

This Phase 1 implementation is designed for:
- **Document Count**: Up to 10,000 document chunks
- **Query Response**: < 2 seconds for most queries  
- **Storage**: Local filesystem only
- **Concurrent Users**: Single user (Claude Desktop)

## Next Steps

Phase 2 improvements could include:
- Distributed vector storage
- Web-based UI
- Multi-user support
- Advanced chunking strategies
- Real-time document updates
