#!/usr/bin/env python3
"""
Integrated RAG MCP Server

This is a standalone MCP server that directly integrates RAG functionality
without requiring a separate HTTP API server. It provides both STDIO (for Claude Desktop)
and HTTP interfaces using a unified architecture.

Usage:
    # STDIO mode (for Claude Desktop):
    python rag_mcp_http_server.py

    # HTTP mode:
    python rag_mcp_http_server.py --mode http --port 8472

MCP Tools Exposed:
    - search_documents: Search the RAG database for relevant documents
    - get_collection_stats: Get statistics about the document collection
    - health_check: Check the health status of the RAG system
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
    import uvicorn
except ImportError as e:
    print(f"Error: FastMCP dependencies not installed. Please install: {e}", file=sys.stderr)
    print("Install with: pip install mcp uvicorn", file=sys.stderr)
    sys.exit(1)

# Import RAG system
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))

try:
    from test_rag_query import RAGQuerySystem
except ImportError:
    print("Error: Cannot import RAGQuerySystem. Make sure tests/test_rag_query.py is available.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("integrated-rag-server")

# Global RAG system instance
rag_system: Optional[RAGQuerySystem] = None

async def initialize_rag_system(chroma_persist_dir: str = "./chroma_db", collection_name: str = "pdf_documents"):
    """Initialize the RAG system globally."""
    global rag_system
    if rag_system is None:
        try:
            logger.info("Initializing RAG system...")
            rag_system = RAGQuerySystem(
                chroma_persist_dir=chroma_persist_dir,
                collection_name=collection_name
            )
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    return rag_system


@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    include_metadata: bool = True
) -> str:
    """Search the RAG database for documents relevant to a query.

    Args:
        query: The search query text
        top_k: Number of results to return (1-50)
        include_metadata: Include document metadata in results

    Returns:
        Formatted search results
    """
    if not query:
        return "Error: Query parameter is required"

    # Ensure RAG system is initialized
    await initialize_rag_system()

    if not rag_system:
        return "Error: RAG system not initialized"

    try:
        start_time = time.time()
        results = rag_system.query(
            query_text=query,
            top_k=top_k,
            include_metadata=include_metadata
        )
        search_time_ms = (time.time() - start_time) * 1000

        return format_search_results(query, results, search_time_ms)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error during search: {str(e)}"


@mcp.tool()
async def get_collection_stats() -> str:
    """Get statistics about the document collection.

    Returns:
        Formatted collection statistics
    """
    # Ensure RAG system is initialized
    await initialize_rag_system()

    if not rag_system:
        return "Error: RAG system not initialized"

    try:
        stats = rag_system.get_collection_stats()
        return format_stats(stats)

    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        return f"Error getting stats: {str(e)}"


@mcp.tool()
async def health_check() -> str:
    """Check the health status of the RAG system.

    Returns:
        Formatted health check results
    """
    # Check if RAG system can be initialized
    try:
        await initialize_rag_system()
    except Exception as e:
        return f"❌ Error: RAG system initialization failed: {str(e)}"

    model_loaded = rag_system is not None
    collection_count = 0

    if model_loaded:
        try:
            stats = rag_system.get_collection_stats()
            collection_count = stats.get('total_documents', 0)
        except Exception:
            pass

    status_emoji = "✅" if model_loaded else "❌"
    model_emoji = "✅" if model_loaded else "❌"

    return f"""RAG System Health Check
{status_emoji} Status: {"healthy" if model_loaded else "not initialized"}
{model_emoji} Model Loaded: {model_loaded}
📊 Document Count: {collection_count}
🕐 Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}"""


def format_search_results(query: str, results: List[Dict[str, Any]], search_time_ms: float) -> str:
    """Format search results for display."""
    if not results:
        return f"No results found for query: '{query}'"

    formatted = f"""Search Results for: "{query}"
Found {len(results)} results in {search_time_ms:.1f}ms

"""

    for result in results:
        rank = result["rank"]
        similarity = result["similarity"]
        document = result["document"]
        metadata = result.get("metadata", {})

        # Truncate long documents
        display_text = document
        if len(display_text) > 500:
            display_text = display_text[:500] + "..."

        formatted += f"Result {rank} (Similarity: {similarity:.3f})\n"

        if metadata:
            filename = metadata.get("filename", "Unknown")
            page_num = metadata.get("page_number", "Unknown")
            formatted += f"Source: {filename} (Page {page_num})\n"

        formatted += f"Text: {display_text}\n"
        formatted += "-" * 80 + "\n\n"

    return formatted.strip()


def format_stats(stats: Dict[str, Any]) -> str:
    """Format collection statistics for display."""
    formatted = f"""Collection Statistics
📊 Total Documents: {stats["total_documents"]}
📁 Collection Name: {stats["collection_name"]}
💾 Storage Directory: {stats["persist_directory"]}
"""

    if stats.get("unique_files"):
        formatted += f"📄 Unique Files: {stats['unique_files']}\n"

    if stats.get("all_filenames"):
        formatted += "\nFiles in Collection:\n"
        for filename in stats["all_filenames"]:
            formatted += f"  • {filename}\n"

    return formatted.strip()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated RAG MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # STDIO mode for Claude Desktop (default)
    python rag_mcp_http_server.py

    # HTTP mode for web access
    python rag_mcp_http_server.py --mode http --port 8472

    # Custom ChromaDB location
    python rag_mcp_http_server.py --chroma-dir ./custom_chroma_db
        """
    )

    parser.add_argument(
        "--mode",
        choices=["stdio", "http"],
        default="stdio",
        help="Server mode: stdio (for Claude Desktop) or http (web server) - default: stdio"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8472,
        help="Port for HTTP server (default: 8472)"
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db",
        help="ChromaDB persistence directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection-name",
        default="pdf_documents",
        help="ChromaDB collection name (default: pdf_documents)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate ChromaDB directory
    chroma_path = Path(args.chroma_dir)
    if not chroma_path.exists():
        logger.error(f"ChromaDB directory not found: {chroma_path}")
        logger.error("Please run init_chroma.py first or use --chroma-dir to specify the correct path")
        sys.exit(1)

    try:
        if args.mode == "stdio":
            # STDIO mode for Claude Desktop
            logger.info("Starting Integrated RAG MCP Server in STDIO mode")
            logger.info(f"ChromaDB directory: {chroma_path.absolute()}")
            logger.info(f"Collection name: {args.collection_name}")

            # Initialize RAG system with custom parameters
            async def run_stdio():
                await initialize_rag_system(args.chroma_dir, args.collection_name)
                await mcp.run()

            asyncio.run(run_stdio())

        elif args.mode == "http":
            # HTTP mode for web access
            logger.info(f"Starting Integrated RAG MCP Server in HTTP mode on {args.host}:{args.port}")
            logger.info(f"ChromaDB directory: {chroma_path.absolute()}")
            logger.info(f"Collection name: {args.collection_name}")

            # Initialize RAG system before starting HTTP server
            async def setup():
                await initialize_rag_system(args.chroma_dir, args.collection_name)

            # Run setup
            asyncio.run(setup())

            # Start HTTP server using FastMCP's run method
            mcp.run(host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()