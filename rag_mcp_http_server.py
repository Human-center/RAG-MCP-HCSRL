#!/usr/bin/env python3
"""
RAG MCP HTTP Server - Public Access

This creates a public HTTP MCP server using FastMCP.
It communicates with the persistent RAG API server via HTTP requests
and exposes MCP tools through proper HTTP transport.

Usage:
    python rag_mcp_http_server.py

Access via:
    http://localhost:8475

MCP Tools Exposed:
    - search_documents: Search the RAG database for relevant documents
    - get_collection_stats: Get statistics about the document collection
    - health_check: Check the health status of the RAG system
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional

try:
    import httpx
    from mcp.server.fastmcp import FastMCP
    import uvicorn
except ImportError as e:
    print(f"Error: Dependencies not installed. Please install: {e}", file=sys.stderr)
    print("Install with: pip install mcp uvicorn httpx", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAG API Server configuration
RAG_API_BASE_URL = "http://127.0.0.1:8921"
RAG_API_TIMEOUT = 30.0

# Initialize FastMCP server
mcp = FastMCP("rag-http-server")

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None

def get_http_client():
    """Get or initialize HTTP client for RAG API communication."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=RAG_API_TIMEOUT)
    return http_client


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

    try:
        search_payload = {
            "query": query,
            "top_k": top_k,
            "include_metadata": include_metadata
        }

        client = get_http_client()
        response = await client.post(
            f"{RAG_API_BASE_URL}/search",
            json=search_payload
        )
        response.raise_for_status()

        result = response.json()
        return format_search_results(result)

    except httpx.RequestError as e:
        logger.error(f"Request to RAG API failed: {e}")
        return f"Error: Could not connect to RAG API server at {RAG_API_BASE_URL}"
    except httpx.HTTPStatusError as e:
        logger.error(f"RAG API returned error: {e.response.status_code}")
        return f"Error: RAG API returned status {e.response.status_code}"
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error during search: {str(e)}"


@mcp.tool()
async def get_collection_stats() -> str:
    """Get statistics about the document collection.

    Returns:
        Formatted collection statistics
    """
    try:
        client = get_http_client()
        response = await client.get(f"{RAG_API_BASE_URL}/stats")
        response.raise_for_status()

        stats = response.json()
        return format_stats(stats)

    except httpx.RequestError as e:
        logger.error(f"Request to RAG API failed: {e}")
        return f"Error: Could not connect to RAG API server at {RAG_API_BASE_URL}"
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        return f"Error getting stats: {str(e)}"


@mcp.tool()
async def health_check() -> str:
    """Check the health status of the RAG system.

    Returns:
        Formatted health check results
    """
    try:
        client = get_http_client()
        response = await client.get(f"{RAG_API_BASE_URL}/health")
        response.raise_for_status()

        health = response.json()

        status_emoji = "✅" if health["status"] == "healthy" else "❌"
        model_emoji = "✅" if health["model_loaded"] else "❌"

        return f"""RAG System Health Check
{status_emoji} Status: {health["status"]}
{model_emoji} Model Loaded: {health["model_loaded"]}
📊 Document Count: {health["collection_count"]}
🕐 Timestamp: {health["timestamp"]}"""

    except httpx.RequestError as e:
        logger.error(f"Request to RAG API failed: {e}")
        return f"❌ Error: Could not connect to RAG API server at {RAG_API_BASE_URL}"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return f"❌ Error during health check: {str(e)}"


def format_search_results(result: Dict[str, Any]) -> str:
    """Format search results for display."""
    query = result["query"]
    results = result["results"]
    total_results = result["total_results"]
    search_time = result["search_time_ms"]

    if not results:
        return f"No results found for query: '{query}'"

    formatted = f"""Search Results for: "{query}"
Found {total_results} results in {search_time:.1f}ms

"""

    for doc_result in results:
        rank = doc_result["rank"]
        similarity = doc_result["similarity"]
        document = doc_result["document"]
        metadata = doc_result.get("metadata", {})

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


async def cleanup():
    """Cleanup resources on shutdown."""
    global http_client
    if http_client:
        await http_client.aclose()


def main():
    """Main entry point."""
    try:
        # Initialize HTTP client (we'll do this in the tool functions)
        logger.info("Starting RAG MCP HTTP Server on http://localhost:8475")
        logger.info(f"RAG API URL: {RAG_API_BASE_URL}")

        # Get the FastAPI app from FastMCP and run it with uvicorn
        # Use streamable_http_app for proper MCP-over-HTTP protocol support
        app = mcp.streamable_http_app()
        
        uvicorn.run(app, host="127.0.0.1", port=8475)

    except KeyboardInterrupt:
        logger.info("HTTP MCP server shutdown requested")
    except Exception as e:
        logger.error(f"HTTP MCP server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()