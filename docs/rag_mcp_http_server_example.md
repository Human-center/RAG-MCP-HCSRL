# Integrated RAG MCP Server Example

This example demonstrates the integrated RAG MCP server with both STDIO and HTTP transport modes, featuring document search, collection statistics, and health monitoring.

## Quick Start

### Prerequisites
Ensure you have initialized your ChromaDB with documents:
```bash
# Initialize ChromaDB
python init_chroma.py

# Ingest PDFs
python ingest_pdfs.py --input-dir ./documents
```

### Run STDIO Mode (Default)
For Claude Desktop integration:
```bash
python rag_mcp_http_server.py
```

### Run HTTP Mode
For web API access:
```bash
python rag_mcp_http_server.py --mode http --port 8472
```

## Custom Configuration

### Custom Port (HTTP Mode)
```bash
python rag_mcp_http_server.py --mode http --port 9000
```

### Custom ChromaDB Location
```bash
python rag_mcp_http_server.py --chroma-dir ./my_custom_db --collection-name my_docs
```

### Debug Logging
```bash
python rag_mcp_http_server.py --verbose
```

### Combined Options
```bash
python rag_mcp_http_server.py --mode http --port 9000 --verbose --chroma-dir ./custom_db
```

## Tools Available

- `search_documents(query, top_k=5, include_metadata=True)` - Search RAG database for relevant documents
- `get_collection_stats()` - Get statistics about document collection  
- `health_check()` - Check RAG system health status

## HTTP Endpoints (HTTP Mode Only)

When running in HTTP mode, the following REST endpoints are available:

- `POST /search` - Search documents
- `GET /health` - Health check
- `GET /stats` - Collection statistics

### Example HTTP Usage

```bash
# Search documents
curl -X POST http://localhost:8472/search \
  -H "Content-Type: application/json" \
  -d '{"query": "protein folding", "top_k": 3, "include_metadata": true}'

# Health check
curl http://localhost:8472/health

# Collection stats
curl http://localhost:8472/stats
```

## Configuration Options

- `--mode` - Server mode: `stdio` (default) or `http`
- `--host` - HTTP server host (default: 127.0.0.1)
- `--port` - HTTP server port (default: 8472)
- `--chroma-dir` - ChromaDB directory (default: ./chroma_db)
- `--collection-name` - Collection name (default: pdf_documents)
- `--verbose` - Enable debug logging

## Claude Desktop Integration

For STDIO mode with Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag_mcp_http_server.py"],
      "cwd": "/path/to/project",
      "env": {
        "PATH": "/path/to/venv/bin:/usr/bin:/bin"
      }
    }
  }
}
```