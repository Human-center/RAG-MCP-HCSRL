# ChromaDB Local Database Implementation

This directory contains a complete ChromaDB local database implementation for the Phase 1 RAG system, as specified in `CLAUDE.md`. The implementation provides persistent vector storage for PDF document embeddings with metadata support and similarity search capabilities.

## 📁 Implementation Files

### Core Implementation
- **`chroma_db.py`** - Main ChromaDB manager class with full CRUD operations
- **`init_chroma.py`** - Database initialization script
- **`requirements.txt`** - All necessary dependencies

### Testing and Examples
- **`test_chroma_db.py`** - Comprehensive test suite for ChromaDB functionality
- **`example_embeddings_integration.py`** - Integration example with EmbeddingGemma

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install core ChromaDB dependencies
pip install chromadb sentence-transformers numpy pandas python-dotenv

# For EmbeddingGemma integration (optional)
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
```

### 2. Initialize Database

```bash
# Initialize with default settings
python init_chroma.py

# Initialize with custom options
python init_chroma.py --persist-dir ./my_db/ --collection-name my_docs

# Reset existing database (WARNING: deletes all data)
python init_chroma.py --reset
```

### 3. Basic Usage

```python
from chroma_db import ChromaDBManager

# Initialize database manager
db_manager = ChromaDBManager()
db_manager.initialize_db()

# Add documents with embeddings and metadata
success = db_manager.add_documents(
    embeddings=your_embeddings,
    documents=your_texts,
    metadatas=[{"filename": "doc.pdf", "page_number": 1}]
)

# Similarity search
results = db_manager.similarity_search(
    query_embedding=query_vector,
    n_results=5
)

# Filtered search
filtered_results = db_manager.similarity_search(
    query_embedding=query_vector,
    n_results=5,
    where_filter={"filename": "specific_doc.pdf"}
)
```

## 🏗️ Architecture

### ChromaDBManager Class

The main `ChromaDBManager` class provides:

#### **Initialization**
- `__init__(persist_directory, collection_name)` - Set up database paths
- `initialize_db()` - Create/connect to persistent ChromaDB instance

#### **Document Operations**
- `add_documents(embeddings, documents, metadatas, ids)` - Add documents with embeddings
- `delete_documents(ids)` - Remove documents by ID
- `reset_collection()` - Clear entire collection (WARNING: deletes all data)

#### **Search Operations**
- `similarity_search(query_embedding, n_results, where_filter)` - Cosine similarity search
- `get_collection_stats()` - Collection statistics and metadata

#### **Connection Management**
- `close()` - Clean up database connections

### Metadata Schema

Required metadata fields for each document:
```python
{
    "filename": str,      # PDF filename
    "page_number": int,   # Page number in PDF
    # Optional additional fields
    "topic": str,
    "author": str,
    "year": int,
    # ... any other custom metadata
}
```

### Storage Structure

```
./chroma_db/                    # Persistent directory
├── chroma.sqlite3              # Main database file
├── <collection_uuid>/          # Collection-specific data
│   ├── data_level0.bin         # Vector data
│   ├── header.bin              # Metadata
│   └── ...
```

## 🧪 Testing

### Run All Tests

```bash
# Run comprehensive test suite
python test_chroma_db.py

# Run with fresh test collection
python test_chroma_db.py --reset

# Run integration example with mock embeddings
python example_embeddings_integration.py --mock-embeddings
```

### Test Coverage

The test suite covers:
- ✅ Database initialization and connection
- ✅ Document addition with validation
- ✅ Similarity search (basic and filtered)
- ✅ Error handling and input validation
- ✅ Document deletion and collection management
- ✅ Data persistence verification

## 🔧 Configuration

### Environment Variables

For EmbeddingGemma integration:
```bash
export HF_TOKEN="your_huggingface_token"
```

### Default Settings

- **Persist Directory**: `./chroma_db/`
- **Collection Name**: `pdf_documents`
- **Embedding Dimension**: 768 (EmbeddingGemma default)
- **Distance Metric**: Cosine similarity

## 📊 Performance Characteristics

### Storage
- **Local file-based persistence** - No external database required
- **SQLite backend** - Reliable and fast for local usage
- **Automatic data compression** - Efficient storage of embeddings

### Search Performance
- **Cosine similarity** - Fast vector comparisons
- **In-memory search** - Quick results for local collections
- **Metadata filtering** - Efficient filtered searches

### Scalability (Phase 1)
- **Suitable for**: Hundreds to thousands of documents
- **Memory usage**: Proportional to collection size
- **Storage**: Compact binary format for embeddings

## 🔗 Integration Points

### With EmbeddingGemma
```python
# Real embeddings (requires EmbeddingGemma setup)
embedder = EmbeddingGemmaIntegration(mock=False)
embeddings = embedder.create_embeddings(
    documents, 
    task_type="Retrieval-document"
)
```

### With PDF Ingestion
```python
# Add extracted PDF content
db_manager.add_documents(
    embeddings=pdf_embeddings,
    documents=pdf_texts,
    metadatas=pdf_metadata  # filename, page_number, etc.
)
```

### With MCP Server
```python
# Query from MCP server
results = db_manager.similarity_search(
    query_embedding=user_query_embedding,
    n_results=5
)
return format_results_for_mcp(results)
```

## 🚦 Error Handling

The implementation includes robust error handling for:

- **Database connection failures**
- **Invalid input validation** (mismatched lengths, missing metadata)
- **Storage permission errors**
- **Memory limitations**
- **Corrupted data recovery**

## 📈 Phase 1 Limitations & Future Enhancements

### Current Phase 1 Scope ✅
- Local file-based storage
- Single ChromaDB instance
- Basic similarity search with metadata filtering
- Simple CRUD operations
- Persistent storage across sessions

### Future Phases 🚀
- **Distributed vector storage** for scaling
- **Advanced chunking strategies** for large documents  
- **Multi-collection management** for different document types
- **Async operations** for better performance
- **Web-based administration interface**
- **Backup and replication** capabilities

## 📝 Example Workflows

### 1. Research Paper Analysis
```python
# Initialize for research papers
db_manager = ChromaDBManager(collection_name="research_papers")
db_manager.initialize_db()

# Add papers with academic metadata
papers_metadata = [
    {"filename": "nature_2023.pdf", "page_number": 1, 
     "journal": "Nature", "year": 2023, "topic": "AI"},
    # ...
]

# Search for specific research topics
results = db_manager.similarity_search(
    query_embedding=topic_embedding,
    where_filter={"journal": "Nature", "year": 2023}
)
```

### 2. Document Knowledge Base
```python
# Multi-document collection
db_manager = ChromaDBManager(collection_name="knowledge_base")
db_manager.initialize_db()

# Query across all documents
results = db_manager.similarity_search(
    query_embedding=question_embedding,
    n_results=10  # Get diverse results
)

# Extract relevant context for RAG
context_texts = results["documents"]
```

## 🆘 Troubleshooting

### Common Issues

1. **ImportError: chromadb not found**
   ```bash
   pip install chromadb
   ```

2. **Permission denied on ./chroma_db/**
   ```bash
   sudo chown -R $USER:$USER ./chroma_db/
   ```

3. **Database locked errors**
   - Ensure only one process accesses the database
   - Call `db_manager.close()` properly

4. **Memory issues with large collections**
   - Use filtered searches to reduce result sets
   - Consider chunking large documents

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

This implementation follows the specifications in `CLAUDE.md` for the Phase 1 RAG system. For issues or enhancements:

1. Check the comprehensive test suite results
2. Review error logs in stderr output  
3. Verify all dependencies are installed correctly
4. Ensure proper file permissions on persist directory

The ChromaDB local database implementation is production-ready for Phase 1 requirements and provides a solid foundation for future scaling and enhancements.