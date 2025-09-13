#!/bin/bash

# Script to start the RAG HTTP MCP Server automatically
# This script activates the virtual environment and starts the server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RAG HTTP MCP Server Startup Script ===${NC}"
echo -e "Working directory: ${SCRIPT_DIR}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment 'venv' not found!${NC}"
    echo -e "Please create a virtual environment first:"
    echo -e "  python3 -m venv venv"
    echo -e "  source venv/bin/activate"
    echo -e "  pip install -r requirements.txt"
    exit 1
fi

# Check if the HTTP server file exists
if [ ! -f "rag_mcp_http_server.py" ]; then
    echo -e "${RED}Error: rag_mcp_http_server.py not found!${NC}"
    exit 1
fi

# Check if ChromaDB directory exists
if [ ! -d "chroma_db" ]; then
    echo -e "${YELLOW}Warning: chroma_db directory not found!${NC}"
    echo -e "You may need to run the ingestion script first:"
    echo -e "  python ingest_pdfs.py --input-dir ./documents"
fi

# Default port for HTTP mode
PORT=${1:-8472}

echo -e "${BLUE}Starting RAG HTTP MCP Server...${NC}"
echo -e "Mode: HTTP"
echo -e "Port: ${PORT}"
echo -e "Time: $(date)"
echo -e "${YELLOW}Server will run in background with nohup${NC}"
echo ""

# Activate virtual environment and start the server
source venv/bin/activate

# Check if required packages are installed
python -c "import fastapi, uvicorn, chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Required packages not installed!${NC}"
    echo -e "Please install requirements:"
    echo -e "  pip install fastapi uvicorn chromadb"
    exit 1
fi

# Start the server with nohup
LOG_FILE="server.log"
PID_FILE="server.pid"

echo -e "${GREEN}Server starting in background...${NC}"
echo -e "Log file: ${LOG_FILE}"
echo -e "PID file: ${PID_FILE}"

# Start server with nohup in HTTP mode
nohup python rag_mcp_http_server.py --mode http --port "$PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID to file
echo $SERVER_PID > "$PID_FILE"

echo -e "${GREEN}✅ Server started successfully!${NC}"
echo -e "PID: ${SERVER_PID}"
echo -e "Port: ${PORT}"
echo -e "Log: tail -f ${LOG_FILE}"
echo -e "Stop: kill ${SERVER_PID} or ./stop_server.sh"