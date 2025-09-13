#!/bin/bash

# Script to stop the RAG HTTP MCP Server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PID_FILE="server.pid"

echo -e "${BLUE}=== RAG HTTP MCP Server Stop Script ===${NC}"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}No PID file found. Checking for running processes...${NC}"

    # Look for running Python processes with our server
    PIDS=$(pgrep -f "rag_mcp_http_server.py")
    if [ -z "$PIDS" ]; then
        echo -e "${RED}No running server found.${NC}"
        exit 1
    else
        echo -e "${YELLOW}Found running server processes: $PIDS${NC}"
        for PID in $PIDS; do
            echo -e "Killing PID: $PID"
            kill $PID
        done
        echo -e "${GREEN}✅ Server processes stopped.${NC}"
        exit 0
    fi
fi

# Read PID from file
SERVER_PID=$(cat "$PID_FILE")

if [ -z "$SERVER_PID" ]; then
    echo -e "${RED}Invalid PID file.${NC}"
    exit 1
fi

echo -e "Stopping server with PID: ${SERVER_PID}"

# Check if process is running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    # Try graceful shutdown first
    kill $SERVER_PID

    # Wait a few seconds
    sleep 3

    # Check if it's still running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Graceful shutdown failed, forcing kill...${NC}"
        kill -9 $SERVER_PID
        sleep 1
    fi

    # Final check
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ Failed to stop server${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Server stopped successfully${NC}"
    fi
else
    echo -e "${YELLOW}Process $SERVER_PID not found (already stopped?)${NC}"
fi

# Clean up PID file
rm -f "$PID_FILE"
echo -e "Cleaned up PID file"