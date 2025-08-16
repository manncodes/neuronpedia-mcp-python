#!/usr/bin/env python3
"""
Test script for Neuronpedia MCP server
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx


class MCPClient:
    def __init__(self, command: list[str], cwd: str = None, env: dict = None):
        self.command = command
        self.cwd = cwd
        self.env = env or {}
        self.process = None
        self.request_id = 0

    async def start(self):
        full_env = os.environ.copy()
        full_env.update(self.env)
        
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=full_env
        )

    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("Process not started")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())
        
        return response

    async def close(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()


@asynccontextmanager
async def mcp_client(command: list[str], cwd: str = None, env: dict = None):
    client = MCPClient(command, cwd, env)
    try:
        await client.start()
        yield client
    finally:
        await client.close()


async def test_neuronpedia_mcp():
    """Test the Neuronpedia MCP server"""
    
    env = {
        "NEURONPEDIA_API_KEY": "sk-np-U3UICQiXymVQwY7l6SFbyEeMc0i5NSCjAE1XClDn1WM0"
    }
    
    command = ["uv", "run", "python", "src/neuronpedia_mcp/server.py"]
    cwd = "/mnt/c/Users/MANN PATEL/mcp/neuronpedia-mcp-python"
    
    async with mcp_client(command, cwd, env) as client:
        # Initialize
        init_response = await client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        )
        print("✅ Initialize:", json.dumps(init_response, indent=2))
        
        # Send initialized notification
        await client.send_request("notifications/initialized")
        
        # List tools
        tools_response = await client.send_request("tools/list")
        print("✅ Tools:", json.dumps(tools_response, indent=2))
        
        # Test attribution graph
        print("\n🧪 Testing attribution graph...")
        graph_response = await client.send_request(
            "tools/call",
            {
                "name": "generate_attribution_graph",
                "arguments": {"prompt": "Hello world"}
            }
        )
        print("✅ Attribution Graph:", json.dumps(graph_response, indent=2))


def test_api_directly():
    """Test the Neuronpedia API directly"""
    print("🧪 Testing Neuronpedia API directly...")
    
    import httpx
    
    client = httpx.Client(
        headers={
            "x-api-key": "sk-np-U3UICQiXymVQwY7l6SFbyEeMc0i5NSCjAE1XClDn1WM0",
            "Content-Type": "application/json"
        }
    )
    
    response = client.post(
        "https://www.neuronpedia.org/api/graph/generate",
        json={"prompt": "test", "modelId": "gemma-2-2b"}
    )
    
    print(f"✅ Status: {response.status_code}")
    print(f"✅ Response: {response.json()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        test_api_directly()
    else:
        asyncio.run(test_neuronpedia_mcp())