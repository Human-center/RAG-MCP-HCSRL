#!/bin/bash

# Script to check the status of the RAG HTTP MCP Server

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
LOG_FILE="server.log"

echo -e "${BLUE}=== RAG HTTP MCP Server Status ===${NC}"

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    SERVER_PID=$(cat "$PID_FILE")
    echo -e "PID file found: ${SERVER_PID}"

    # Check if process is actually running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "Status: ${GREEN}✅ RUNNING${NC}"
        echo -e "PID: ${SERVER_PID}"

        # Get process info
        echo -e "Process info:"
        ps -p $SERVER_PID -o pid,ppid,cmd,etime,pcpu,pmem

        # Try to get server port (look in process command line)
        PORT=$(ps -p $SERVER_PID -o cmd --no-headers | grep -o '\--port [0-9]*' | awk '{print $2}')
        if [ -n "$PORT" ]; then
            echo -e "Port: ${PORT}"
            echo -e "URL: http://localhost:${PORT}"

            # Test if server is responding
            if command -v curl > /dev/null 2>&1; then
                echo -e "\nTesting server response..."
                if curl -s "http://localhost:${PORT}/health" > /dev/null; then
                    echo -e "Health check: ${GREEN}✅ RESPONDING${NC}"
                else
                    echo -e "Health check: ${YELLOW}⚠️  NOT RESPONDING${NC}"
                fi
            fi
        fi

    else
        echo -e "Status: ${RED}❌ NOT RUNNING${NC} (stale PID file)"
        echo -e "Cleaning up stale PID file..."
        rm -f "$PID_FILE"
    fi
else
    echo -e "PID file: ${YELLOW}Not found${NC}"

    # Look for running processes
    PIDS=$(pgrep -f "rag_mcp_http_server.py")
    if [ -n "$PIDS" ]; then
        echo -e "Status: ${YELLOW}⚠️  RUNNING WITHOUT PID FILE${NC}"
        echo -e "Found processes: $PIDS"
        for PID in $PIDS; do
            echo -e "Process $PID:"
            ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem
        done
    else
        echo -e "Status: ${RED}❌ NOT RUNNING${NC}"
    fi
fi

# Show log file info if exists
if [ -f "$LOG_FILE" ]; then
    echo -e "\nLog file: ${LOG_FILE}"
    echo -e "Log size: $(du -h "$LOG_FILE" | cut -f1)"
    echo -e "Last modified: $(stat -f "%Sm" "$LOG_FILE" 2>/dev/null || stat -c "%y" "$LOG_FILE" 2>/dev/null)"
    echo -e "Last 5 lines:"
    tail -n 5 "$LOG_FILE" | sed 's/^/  /'
fi

echo -e "\n${BLUE}Commands:${NC}"
echo -e "  Start:  ./start_server.sh [port]"
echo -e "  Stop:   ./stop_server.sh"
echo -e "  Logs:   tail -f $LOG_FILE"