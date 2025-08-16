#!/usr/bin/env python3

import asyncio
import os
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    ListToolsRequestParams,
    TextContent,
    Tool,
)
from pydantic import BaseModel


class AttributionGraphResponse(BaseModel):
    message: str
    s3url: str
    url: str
    numNodes: int
    numLinks: int


class NeuronpediaClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.neuronpedia.org/api"
        self.client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
        )

    async def generate_attribution_graph(
        self,
        prompt: str,
        model_id: str = "gemma-2-2b",
        max_logits: Optional[int] = None,
        logit_probability: Optional[float] = None,
        node_threshold: Optional[float] = None,
        edge_threshold: Optional[float] = None,
    ) -> AttributionGraphResponse:
        data = {
            "prompt": prompt,
            "modelId": model_id,
        }
        
        if max_logits is not None:
            data["max_logits"] = max_logits
        if logit_probability is not None:
            data["logit_probability"] = logit_probability
        if node_threshold is not None:
            data["node_threshold"] = node_threshold
        if edge_threshold is not None:
            data["edge_threshold"] = edge_threshold

        response = await self.client.post(f"{self.base_url}/graph/generate", json=data)
        response.raise_for_status()
        return AttributionGraphResponse(**response.json())

    async def generate_explanation(self, model: str, layer: int, feature: int) -> dict:
        data = {"model": model, "layer": layer, "feature": feature}
        response = await self.client.post(f"{self.base_url}/explanations/generate", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations(self, query: str, model: Optional[str] = None, layer: Optional[int] = None) -> list:
        params = {"query": query}
        if model:
            params["model"] = model
        if layer is not None:
            params["layer"] = layer
        
        response = await self.client.get(f"{self.base_url}/explanations/search", params=params)
        response.raise_for_status()
        return response.json()

    async def get_activations(self, model: str, layer: int, feature: int, text: str) -> list:
        data = {"model": model, "layer": layer, "feature": feature, "text": text}
        response = await self.client.post(f"{self.base_url}/activations", json=data)
        response.raise_for_status()
        return response.json()

    async def search_top_features(self, model: str, layer: int, text: str, top_k: int = 10) -> dict:
        data = {"model": model, "layer": layer, "text": text, "top_k": top_k}
        response = await self.client.post(f"{self.base_url}/search/top-features", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_generation(
        self, 
        model: str, 
        layer: int, 
        feature: int, 
        prompt: str, 
        steering_strength: float, 
        is_chat: bool = False
    ) -> dict:
        data = {
            "model": model,
            "layer": layer,
            "feature": feature,
            "prompt": prompt,
            "steering_strength": steering_strength,
            "is_chat": is_chat,
        }
        response = await self.client.post(f"{self.base_url}/steering/generate", json=data)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


server = Server("neuronpedia-mcp")
neuronpedia_client: Optional[NeuronpediaClient] = None


def get_client() -> NeuronpediaClient:
    global neuronpedia_client
    if neuronpedia_client is None:
        api_key = os.getenv("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("NEURONPEDIA_API_KEY environment variable is required")
        neuronpedia_client = NeuronpediaClient(api_key)
    return neuronpedia_client


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_attribution_graph",
            description="Generate an attribution graph for analyzing text prompts using Gemma 2-2b",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text prompt to analyze"},
                    "maxLogits": {"type": "number", "description": "Maximum number of logits (optional)"},
                    "logitProbability": {"type": "number", "description": "Logit probability threshold (optional)"},
                    "nodeThreshold": {"type": "number", "description": "Node threshold for graph (optional)"},
                    "edgeThreshold": {"type": "number", "description": "Edge threshold for graph (optional)"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="generate_explanation",
            description="Generate an explanation for a specific feature in an AI model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name (e.g., gpt2-small)"},
                    "layer": {"type": "number", "description": "Layer number"},
                    "feature": {"type": "number", "description": "Feature number"},
                },
                "required": ["model", "layer", "feature"],
            },
        ),
        Tool(
            name="search_explanations",
            description="Search for explanations across models and layers",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "model": {"type": "string", "description": "Optional model filter"},
                    "layer": {"type": "number", "description": "Optional layer filter"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_activations",
            description="Get activation values for a specific feature on given text",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "layer": {"type": "number", "description": "Layer number"},
                    "feature": {"type": "number", "description": "Feature number"},
                    "text": {"type": "string", "description": "Input text to analyze"},
                },
                "required": ["model", "layer", "feature", "text"],
            },
        ),
        Tool(
            name="search_top_features",
            description="Find the top activating features for given text",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "layer": {"type": "number", "description": "Layer number"},
                    "text": {"type": "string", "description": "Input text to analyze"},
                    "topK": {"type": "number", "description": "Number of top features to return (default: 10)"},
                },
                "required": ["model", "layer", "text"],
            },
        ),
        Tool(
            name="steer_generation",
            description="Steer model generation using a specific feature",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name"},
                    "layer": {"type": "number", "description": "Layer number"},
                    "feature": {"type": "number", "description": "Feature number"},
                    "prompt": {"type": "string", "description": "Generation prompt"},
                    "steeringStrength": {"type": "number", "description": "Steering strength (-10 to 10)"},
                    "isChat": {"type": "boolean", "description": "Whether this is a chat model (default: false)"},
                },
                "required": ["model", "layer", "feature", "prompt", "steeringStrength"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        client = get_client()
        
        if name == "generate_attribution_graph":
            result = await client.generate_attribution_graph(
                prompt=arguments["prompt"],
                max_logits=arguments.get("maxLogits"),
                logit_probability=arguments.get("logitProbability"),
                node_threshold=arguments.get("nodeThreshold"),
                edge_threshold=arguments.get("edgeThreshold"),
            )
            
            return [TextContent(
                type="text",
                text=f"Attribution Graph Generated Successfully!\n\n"
                     f"🔗 **View Graph**: {result.url}\n"
                     f"📊 **Nodes**: {result.numNodes}\n"
                     f"🔗 **Links**: {result.numLinks}\n"
                     f"💾 **Data**: {result.s3url}\n\n"
                     f"The graph shows how different parts of the Gemma 2-2b model contribute "
                     f"to generating each token in your prompt: '{arguments['prompt']}'"
            )]
            
        elif name == "generate_explanation":
            result = await client.generate_explanation(
                model=arguments["model"],
                layer=arguments["layer"],
                feature=arguments["feature"]
            )
            return [TextContent(type="text", text=str(result))]
            
        elif name == "search_explanations":
            result = await client.search_explanations(
                query=arguments["query"],
                model=arguments.get("model"),
                layer=arguments.get("layer")
            )
            return [TextContent(type="text", text=str(result))]
            
        elif name == "get_activations":
            result = await client.get_activations(
                model=arguments["model"],
                layer=arguments["layer"],
                feature=arguments["feature"],
                text=arguments["text"]
            )
            return [TextContent(type="text", text=str(result))]
            
        elif name == "search_top_features":
            result = await client.search_top_features(
                model=arguments["model"],
                layer=arguments["layer"],
                text=arguments["text"],
                top_k=arguments.get("topK", 10)
            )
            return [TextContent(type="text", text=str(result))]
            
        elif name == "steer_generation":
            result = await client.steer_generation(
                model=arguments["model"],
                layer=arguments["layer"],
                feature=arguments["feature"],
                prompt=arguments["prompt"],
                steering_strength=arguments["steeringStrength"],
                is_chat=arguments.get("isChat", False)
            )
            return [TextContent(type="text", text=str(result))]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="neuronpedia-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())