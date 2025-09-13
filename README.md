# RAG Database with Model Context Protocol (MCP) Server

A comprehensive Retrieval-Augmented Generation (RAG) system that integrates with AI assistants like Claude, ChatGPT, and Gemini through the Model Context Protocol (MCP). This system allows you to ingest PDF documents, create vector embeddings, and query your document collection using natural language.

## 🎯 Overview

This RAG system consists of three main components:

1. **Document Ingestion Pipeline** - Extracts text from PDFs and creates vector embeddings
2. **Vector Database** - Local ChromaDB storage with persistent file-based storage
3. **MCP Server** - Exposes the RAG database through standardized tools for AI assistants

### Key Features

- 📄 **PDF Document Processing** - Automatic text extraction and chunking
- 🔍 **Semantic Search** - Vector similarity search using embeddings
- 🤖 **AI Assistant Integration** - Works with Claude Desktop, ChatGPT, and other MCP-compatible clients
- 💾 **Local Storage** - All data stored locally, no external dependencies
- ⚡ **Fast Performance** - Persistent API server keeps models loaded
- 🔧 **Easy Setup** - Automated installation and configuration scripts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- 4GB+ RAM (for embedding models)
- macOS, Linux, or Windows

### 1. Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd RAG-MCP-HCSRL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run automated setup (installs dependencies, creates directories, tests functionality)
python setup.py
```

### 2. Initialize Database

```bash
# Initialize ChromaDB
python init_chroma.py
```

### 3. Add Documents

```bash
# Create documents directory and add your PDF files
mkdir -p documents
# Copy your PDF files to the documents/ directory

# Ingest PDFs into the database
python ingest_pdfs.py --input-dir ./documents
```

### 4. Start the System

```bash
# Terminal 1: Start the RAG API server (keeps models loaded)
source venv/bin/activate
python rag_api_server.py --port 8000

# Terminal 2: Test the system
python tests/test_rag_query.py --query "your search query here"
```

## 🔧 AI Assistant Integration

### Claude Desktop Integration

Claude Desktop uses the MCP protocol to connect to your RAG database.

#### Configuration

1. **Configure Claude Desktop MCP Settings**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent on other platforms:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/full/path/to/RAG-MCP-HCSRL/rag_mcp_server.py"],
      "cwd": "/full/path/to/RAG-MCP-HCSRL",
      "env": {
        "PATH": "/full/path/to/RAG-MCP-HCSRL/venv/bin:/usr/bin:/bin"
      }
    }
  }
}
```

2. **Start Required Services**

```bash
# Start the RAG API server first
source venv/bin/activate && python rag_api_server.py --port 8000
```

3. **Restart Claude Desktop**

The MCP server will automatically start when Claude Desktop launches.

#### Usage in Claude Desktop

Once configured, you can ask Claude to search your documents:

```
Search my documents for information about protein folding
Find papers related to machine learning in my database
What does my collection say about renewable energy?
```

### ChatGPT Integration

ChatGPT can access your RAG system through API calls or custom GPT actions.

#### Option 1: API Integration

```bash
# Start the RAG API server
python rag_api_server.py --host 0.0.0.0 --port 8000

# ChatGPT can then make HTTP requests to your server
# POST http://your-server:8000/search
# Body: {"query": "search terms", "top_k": 5}
```

#### Option 2: Custom GPT Actions

Create a custom GPT with these action definitions:

```yaml
openapi: 3.0.1
info:
  title: RAG Document Search
  version: 1.0.0
servers:
  - url: http://your-server:8000
paths:
  /search:
    post:
      summary: Search documents
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: Search query
                top_k:
                  type: integer
                  description: Number of results
      responses:
        '200':
          description: Search results
```

### Google Gemini Integration

Gemini can integrate through function calling or extensions.

#### Function Calling Setup

```python
# Example Gemini function definition
search_documents_function = {
    "name": "search_documents",
    "description": "Search RAG document database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"}
        },
        "required": ["query"]
    }
}
```

## 📁 System Architecture

```
RAG-MCP-HCSRL/
├── README.md                 # This file
├── CLAUDE.md                # Project instructions for Claude
├── requirements.txt         # Python dependencies
├── setup.py                # System setup and verification
├── init_chroma.py          # Database initialization
├── chroma_db.py            # Core database manager
├── ingest_pdfs.py          # PDF ingestion pipeline
├── rag_api_server.py       # Persistent API server
├── rag_mcp_server.py       # MCP protocol server
├── documents/              # PDF documents directory
├── chroma_db/              # ChromaDB storage (created automatically)
├── tests/                  # Test scripts
│   ├── test_rag_query.py   # RAG query testing
│   └── test_chroma_db.py   # Database testing
└── venv/                   # Virtual environment
```

## 🛠️ Detailed Setup Guide

### Manual Installation

If the automated setup doesn't work, follow these manual steps:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -U sentence-transformers
pip install torch torchvision torchaudio
pip install chromadb python-dotenv
pip install fastapi uvicorn "mcp[cli]"
pip install pypdf2 pymupdf python-multipart
pip install httpx numpy pandas tqdm nltk

# 3. Create directories
mkdir -p chroma_db documents

# 4. Initialize database
python init_chroma.py

# 5. Test installation
python tests/test_chroma_db.py
```

### Environment Configuration

Create a `.env` file for custom configuration:

```env
# .env file
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=pdf_documents
API_PORT=8000
API_HOST=127.0.0.1
LOG_LEVEL=INFO
```

## 📊 Usage Examples

### Command Line Usage

```bash
# Search documents
python tests/test_rag_query.py --query "machine learning applications"

# Get collection statistics
python -c "
from tests.test_rag_query import RAGQuerySystem
rag = RAGQuerySystem()
print(rag.get_collection_stats())
"

# Ingest new documents
python ingest_pdfs.py --input-dir ./new_documents --chunk-size 1000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Search documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "protein folding", "top_k": 3}'

# Get statistics
curl http://localhost:8000/stats
```

### MCP Tools Available

When connected through MCP, these tools are available:

- `search_documents` - Search the document database
- `get_collection_stats` - Get database statistics
- `health_check` - Check system health

## ⚙️ Configuration Options

### Database Configuration

```bash
# Custom database location
python init_chroma.py --chroma-dir ./custom_db

# Custom collection name
python init_chroma.py --collection-name research_papers

# Reset database
python init_chroma.py --reset
```

### API Server Configuration

```bash
# Custom host and port
python rag_api_server.py --host 0.0.0.0 --port 8080

# Verbose logging
python rag_api_server.py --verbose

# Custom database location
python rag_api_server.py --chroma-dir ./custom_db
```

### Document Ingestion Options

```bash
# Custom chunk size
python ingest_pdfs.py --input-dir ./docs --chunk-size 500

# Process specific files
python ingest_pdfs.py --input-file ./document.pdf

# Skip existing documents
python ingest_pdfs.py --input-dir ./docs --skip-existing
```

## 🔍 Troubleshooting

### Common Issues

**1. "ChromaDB directory not found"**
```bash
python init_chroma.py  # Initialize the database first
```

**2. "RAG API returned status 500"**
```bash
# Restart the API server
pkill -f rag_api_server.py
python rag_api_server.py --port 8000
```

**3. "Module not found" errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**4. MCP connection issues**
```bash
# Check Claude Desktop config file path and syntax
# Ensure full absolute paths are used
# Restart Claude Desktop after config changes
```

### Performance Optimization

**For better performance:**

1. **Use GPU acceleration** (if available):
   ```bash
   # Install CUDA-compatible PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Increase chunk size** for longer documents:
   ```bash
   python ingest_pdfs.py --chunk-size 1500
   ```

3. **Keep API server running** to avoid model loading delays

## 📈 System Monitoring

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Monitor logs
tail -f rag_api_server.log

# Database statistics
python -c "
from chroma_db import ChromaDBManager
db = ChromaDBManager()
db.initialize_db()
print(db.get_collection_stats())
"
```

### Performance Metrics

- **First search**: ~7 seconds (includes model loading)
- **Subsequent searches**: ~30ms
- **Memory usage**: ~2-4GB (with models loaded)
- **Storage**: ~1MB per 100 pages of documents

## 🔒 Security Considerations

- **Local storage only** - No data sent to external services
- **Network access** - API server runs on localhost by default
- **File permissions** - Ensure proper access controls on document directory
- **API security** - Consider authentication for production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Specify your license here]

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Create an issue in the repository
4. Include system information and error logs

## 🔄 Updates and Maintenance

### Updating the System

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
pkill -f rag_api_server.py
python rag_api_server.py --port 8000
```

### Backup and Restore

```bash
# Backup database
tar -czf chroma_backup.tar.gz chroma_db/

# Restore database
tar -xzf chroma_backup.tar.gz
```

---

**Note**: This system is designed for local use and development. For production deployments, consider additional security measures, monitoring, and scaling configurations.