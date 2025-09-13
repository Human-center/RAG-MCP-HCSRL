#!/usr/bin/env python3
"""
Persistent RAG API Server

This server keeps the RAG model loaded in memory and provides HTTP API endpoints
for performing searches. The model remains loaded between requests for better performance.

Usage:
    python rag_api_server.py --port 8000 --host 127.0.0.1

API Endpoints:
    POST /search - Perform RAG search with query
    GET /stats - Get collection statistics
    GET /health - Health check endpoint
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    print(f"Error: FastAPI dependencies not installed. Please install: {e}", file=sys.stderr)
    print("Install with: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Import our RAG system
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

try:
    from test_rag_query import RAGQuerySystem
except ImportError:
    print("Error: Cannot import RAGQuerySystem. Make sure tests/test_rag_query.py is available.", file=sys.stderr)
    sys.exit(1)


# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    include_metadata: bool = Field(default=True, description="Include metadata in results")


class SearchResult(BaseModel):
    """Model for individual search result."""
    rank: int
    document: str
    distance: float
    similarity: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    total_documents: int
    collection_name: str
    persist_directory: str
    sample_filenames: Optional[List[str]] = None
    unique_files: Optional[int] = None
    all_filenames: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    collection_count: int


class RAGAPIServer:
    """Persistent RAG API Server that keeps the model loaded in memory."""
    
    def __init__(self, chroma_persist_dir: str = "./chroma_db", collection_name: str = "pdf_documents"):
        """Initialize the RAG API server."""
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.rag_system = None
        self.app = FastAPI(
            title="RAG API Server",
            description="Persistent RAG system with loaded model for fast searches",
            version="1.0.0"
        )
        
        # Add CORS middleware for web access
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    async def initialize_rag_system(self):
        """Initialize the RAG system and keep it loaded."""
        try:
            logger.info("Initializing RAG system...")
            self.rag_system = RAGQuerySystem(
                chroma_persist_dir=self.chroma_persist_dir,
                collection_name=self.collection_name
            )
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize RAG system on startup."""
            await self.initialize_rag_system()
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search(request: SearchRequest):
            """Perform RAG search."""
            if not self.rag_system:
                raise HTTPException(status_code=503, detail="RAG system not initialized")
            
            start_time = time.time()
            try:
                results = self.rag_system.query(
                    query_text=request.query,
                    top_k=request.top_k,
                    include_metadata=request.include_metadata
                )
                search_time_ms = (time.time() - start_time) * 1000
                
                # Convert results to response format
                search_results = []
                for result in results:
                    search_results.append(SearchResult(
                        rank=result['rank'],
                        document=result['document'],
                        distance=result['distance'],
                        similarity=result['similarity'],
                        metadata=result.get('metadata')
                    ))
                
                return SearchResponse(
                    query=request.query,
                    results=search_results,
                    total_results=len(search_results),
                    search_time_ms=search_time_ms
                )
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """Get collection statistics."""
            if not self.rag_system:
                raise HTTPException(status_code=503, detail="RAG system not initialized")
            
            try:
                stats = self.rag_system.get_collection_stats()
                return StatsResponse(
                    total_documents=stats.get('total_documents', 0),
                    collection_name=stats.get('collection_name', ''),
                    persist_directory=stats.get('persist_directory', ''),
                    sample_filenames=stats.get('sample_filenames'),
                    unique_files=stats.get('unique_files'),
                    all_filenames=stats.get('all_filenames')
                )
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            model_loaded = self.rag_system is not None
            collection_count = 0
            
            if model_loaded:
                try:
                    stats = self.rag_system.get_collection_stats()
                    collection_count = stats.get('total_documents', 0)
                except:
                    pass
            
            return HealthResponse(
                status="healthy" if model_loaded else "initializing",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_loaded=model_loaded,
                collection_count=collection_count
            )
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "RAG API Server",
                "version": "1.0.0",
                "endpoints": {
                    "POST /search": "Perform RAG search",
                    "GET /stats": "Get collection statistics",
                    "GET /health": "Health check",
                    "GET /docs": "API documentation"
                }
            }


def main():
    """Main entry point for RAG API server."""
    parser = argparse.ArgumentParser(
        description="Persistent RAG API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server on default port 8000
    python rag_api_server.py
    
    # Start on custom port and host
    python rag_api_server.py --port 8080 --host 0.0.0.0
    
    # Use custom ChromaDB location
    python rag_api_server.py --chroma-dir ./custom_chroma_db
    
    # Test the server:
    curl -X POST "http://localhost:8000/search" \
         -H "Content-Type: application/json" \
         -d '{"query": "protein folding", "top_k": 3}'
        """
    )
    
    # Server options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
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
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        args.log_level = "DEBUG"
    
    logger.setLevel(getattr(logging, args.log_level))
    
    # Validate ChromaDB directory
    chroma_path = Path(args.chroma_dir)
    if not chroma_path.exists():
        logger.error(f"ChromaDB directory not found: {chroma_path}")
        logger.error("Please run init_chroma.py first or use --chroma-dir to specify the correct path")
        sys.exit(1)
    
    try:
        # Create and configure the RAG API server
        logger.info(f"Starting RAG API Server on {args.host}:{args.port}")
        logger.info(f"ChromaDB directory: {chroma_path.absolute()}")
        logger.info(f"Collection name: {args.collection_name}")
        
        server = RAGAPIServer(
            chroma_persist_dir=args.chroma_dir,
            collection_name=args.collection_name
        )
        
        # Run the server
        uvicorn.run(
            server.app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()