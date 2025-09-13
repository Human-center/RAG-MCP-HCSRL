#!/usr/bin/env python3
"""
RAG MCP Server - STDIO Interface

This MCP server provides a STDIO interface to the RAG API server.
It communicates with the persistent RAG API server via HTTP requests
and exposes MCP tools for Claude Desktop integration.

Usage:
    python rag_mcp_server.py

MCP Tools Exposed:
    - search_documents: Search the RAG database for relevant documents
    - get_collection_stats: Get statistics about the document collection
    - health_check: Check the health status of the RAG system
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

try:
    import httpx
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
except ImportError as e:
    print(f"Error: MCP dependencies not installed. Please install: {e}", file=sys.stderr)
    print("Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Configure logging to stderr (never stdout for STDIO servers)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# RAG API Server configuration
RAG_API_BASE_URL = "http://127.0.0.1:8000"
RAG_API_TIMEOUT = 30.0


class RAGMCPServer:
    """MCP Server that interfaces with the RAG API server."""
    
    def __init__(self, rag_api_url: str = RAG_API_BASE_URL):
        """Initialize the MCP server."""
        self.rag_api_url = rag_api_url
        self.server = Server("rag-mcp-server")
        self.http_client = None
        
        # Register MCP tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools with the server."""
        
        @self.server.list_tools()
        async def handle_list_tools():
            """List available MCP tools."""
            return [
                Tool(
                    name="search_documents",
                    description="Search the RAG database for documents relevant to a query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (1-50)",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 5
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Include document metadata in results",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_collection_stats",
                    description="Get statistics about the document collection",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                ),
                Tool(
                    name="health_check",
                    description="Check the health status of the RAG system",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool execution requests."""
            try:
                if name == "search_documents":
                    return await self._search_documents(arguments)
                elif name == "get_collection_stats":
                    return await self._get_collection_stats()
                elif name == "health_check":
                    return await self._health_check()
                else:
                    raise ValueError(f"Unknown tool: {name}")
            
            except Exception as e:
                logger.error(f"Tool execution failed for {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def _search_documents(self, arguments: dict) -> List[TextContent]:
        """Execute document search via RAG API."""
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        include_metadata = arguments.get("include_metadata", True)
        
        if not query:
            return [TextContent(
                type="text",
                text="Error: Query parameter is required"
            )]
        
        try:
            # Make request to RAG API
            search_payload = {
                "query": query,
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            
            response = await self.http_client.post(
                f"{self.rag_api_url}/search",
                json=search_payload,
                timeout=RAG_API_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Format results for MCP response
            formatted_response = self._format_search_results(result)
            
            return [TextContent(
                type="text",
                text=formatted_response
            )]
            
        except httpx.RequestError as e:
            logger.error(f"Request to RAG API failed: {e}")
            return [TextContent(
                type="text",
                text=f"Error: Could not connect to RAG API server at {self.rag_api_url}. Make sure it's running."
            )]
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG API returned error: {e.response.status_code}")
            return [TextContent(
                type="text",
                text=f"Error: RAG API returned status {e.response.status_code}"
            )]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [TextContent(
                type="text",
                text=f"Error during search: {str(e)}"
            )]
    
    async def _get_collection_stats(self) -> List[TextContent]:
        """Get collection statistics via RAG API."""
        try:
            response = await self.http_client.get(
                f"{self.rag_api_url}/stats",
                timeout=RAG_API_TIMEOUT
            )
            response.raise_for_status()
            
            stats = response.json()
            
            # Format stats for display
            formatted_stats = self._format_stats(stats)
            
            return [TextContent(
                type="text",
                text=formatted_stats
            )]
            
        except httpx.RequestError as e:
            logger.error(f"Request to RAG API failed: {e}")
            return [TextContent(
                type="text",
                text=f"Error: Could not connect to RAG API server at {self.rag_api_url}"
            )]
        except Exception as e:
            logger.error(f"Stats request failed: {e}")
            return [TextContent(
                type="text",
                text=f"Error getting stats: {str(e)}"
            )]
    
    async def _health_check(self) -> List[TextContent]:
        """Check RAG system health via API."""
        try:
            response = await self.http_client.get(
                f"{self.rag_api_url}/health",
                timeout=RAG_API_TIMEOUT
            )
            response.raise_for_status()
            
            health = response.json()
            
            # Format health info
            status_emoji = "✅" if health["status"] == "healthy" else "❌"
            model_emoji = "✅" if health["model_loaded"] else "❌"
            
            formatted_health = f"""RAG System Health Check
{status_emoji} Status: {health["status"]}
{model_emoji} Model Loaded: {health["model_loaded"]}
📊 Document Count: {health["collection_count"]}
🕐 Timestamp: {health["timestamp"]}"""
            
            return [TextContent(
                type="text",
                text=formatted_health
            )]
            
        except httpx.RequestError as e:
            logger.error(f"Request to RAG API failed: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Error: Could not connect to RAG API server at {self.rag_api_url}"
            )]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Error during health check: {str(e)}"
            )]
    
    def _format_search_results(self, result: dict) -> str:
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
        
        for i, doc_result in enumerate(results, 1):
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
    
    def _format_stats(self, stats: dict) -> str:
        """Format collection statistics for display."""
        formatted = f"""Collection Statistics
📊 Total Documents: {stats["total_documents"]}
📁 Collection Name: {stats["collection_name"]}
💾 Storage Directory: {stats["persist_directory"]}
"""
        
        if stats.get("unique_files"):
            formatted += f"📄 Unique Files: {stats["unique_files"]}\n"
        
        if stats.get("all_filenames"):
            formatted += "\nFiles in Collection:\n"
            for filename in stats["all_filenames"]:
                formatted += f"  • {filename}\n"
        
        return formatted.strip()
    
    async def start(self):
        """Start the MCP server."""
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient()
        
        # Test connection to RAG API
        try:
            response = await self.http_client.get(
                f"{self.rag_api_url}/health",
                timeout=5.0
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to RAG API at {self.rag_api_url}")
        except Exception as e:
            logger.error(f"Failed to connect to RAG API at {self.rag_api_url}: {e}")
            logger.error("Make sure the RAG API server is running before starting the MCP server")
            return
        
        # Run STDIO server
        logger.info("Starting RAG MCP Server with STDIO transport")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()


async def main():
    """Main entry point for the MCP server."""
    try:
        # Create and start the MCP server
        mcp_server = RAGMCPServer()
        await mcp_server.start()
        
    except KeyboardInterrupt:
        logger.info("MCP server shutdown requested")
    except Exception as e:
        logger.error(f"MCP server failed: {e}")
        sys.exit(1)
    finally:
        if 'mcp_server' in locals():
            await mcp_server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())