Directory structure:
└── jonigl-mcp-server-with-streamable-http-example/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    └── simple_streamable_http_mcp_server.py


Files Content:

================================================
FILE: README.md
================================================
# A simple MCP server with streamable HTTP transport Example
This example demonstrates how to create a simple MCP server with streamable HTTP transport, featuring several tools, prompts, and resources.

## Run

By default, the server runs on port **8000**.

```bash
python simple_streamable_http_mcp_server.py
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv run mcp-server
```

## Custom Port

Change the port (default is 8000):

```bash
MCP_SERVER_PORT=9000 python simple_streamable_http_mcp_server.py
```

## Debug Logging

Enable debug logs for tool calls:

```bash
MCP_DEBUG=1 python simple_streamable_http_mcp_server.py
```

## Both Together

```bash
MCP_SERVER_PORT=9000 MCP_DEBUG=1 python simple_streamable_http_mcp_server.py
```

## Tools
- `hello_world(name)` - Say hello
- `add_numbers(a, b)` - Add two numbers
- `random_number(min_val, max_val)` - Generate random number
- `return_json_example()` - Return example JSON
- `calculate_bmi(weight, height)` - Calculate BMI

## Prompts
- `BMI Calculator` - Prompt for BMI calculation

## Resources
- `server://info` - Get server info



================================================
FILE: pyproject.toml
================================================
[project]
name = "stream-mcp-server-test"
version = "0.1.0"
description = "Streamable MCP Server Test with Random Number Generation"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "mcp>=1.0.0",
]

[project.scripts]
mcp-server = "simple_streamable_http_mcp_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]



================================================
FILE: requirements.txt
================================================
fastapi
uvicorn[standard]
mcp



================================================
FILE: simple_streamable_http_mcp_server.py
================================================
#!/usr/bin/env python3
"""
Simple MCP Server with Streamable HTTP Transport
"""

import os
import random
from mcp.server.fastmcp import FastMCP

# configurable port by environment variable
port = int(os.environ.get("MCP_SERVER_PORT", 8000))

# Create a basic stateless MCP server
mcp = FastMCP(name="Simple MCP Server with Streamable HTTP Transport", port=port, stateless_http=True)

# Add debug logging flag based on environment variable
DEBUG = os.environ.get("MCP_DEBUG", "0").lower() in ("1", "true", "yes")

@mcp.tool()
def hello_world(name: str = "World") -> str:
    """Say hello to someone"""
    result = f"Hello, {name}!"
    if DEBUG:
        print(f"[DEBUG] hello_world called with name={name} -> {result}")
    return result

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together"""
    result = a + b
    if DEBUG:
        print(f"[DEBUG] add_numbers called with a={a}, b={b} -> {result}")
    return result

@mcp.tool()
def random_number(min_val: int = 0, max_val: int = 100) -> int:
    """Generate a random integer between min_val and max_val (inclusive)"""
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    result = random.randint(min_val, max_val)
    if DEBUG:
        print(f"[DEBUG] random_number called with min_val={min_val}, max_val={max_val} -> {result}")
    return result

@mcp.tool()
def return_json_example() -> dict:
    """Return a JSON example"""
    result = {"message": "This is a JSON response", "status": "success"}
    if DEBUG:
        print(f"[DEBUG] return_json_example called -> {result}")
    return result

@mcp.tool()
def calculate_bmi(weight: float, height: float) -> str:
    """Calculate BMI from weight and height"""
    bmi = weight / (height ** 2)
    result = f"Your BMI is {bmi:.2f}"
    if DEBUG:
        print(f"[DEBUG] calculate_bmi called with weight={weight}, height={height} -> {result}")
    return result

@mcp.resource("server://info")
async def get_server_info() -> str:
    """Get information about this server"""
    return "This is a simple MCP server with streamable HTTP transport. It supports tools for greeting, adding numbers, generating random numbers, and calculating BMI. It also provides a BMI calculator prompt."

@mcp.prompt(title="BMI Calculator", description="Calculate BMI from weight and height")
def prompt_bmi_calculator(weight: float, height: float) -> str:
    """Prompt for BMI calculation"""
    return f"Please calculate my BMI using the following information: Weight: {weight} kg, Height: {height} m."

def main():
    """Main entry point for the MCP server"""
    print("Starting Simple MCP Server...")
    print("Available tools: hello_world, add_numbers, random_number, calculate_bmi")
    print("Available prompts: BMI Calculator")
    print("Available resources: server://info")

    # Run with streamable HTTP transport
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()


