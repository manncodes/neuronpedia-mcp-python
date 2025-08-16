#!/usr/bin/env python3

import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
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

    async def get_activations(self, model_id: str, layer: int, index: int, text: str) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index, "text": text}
        response = await self.client.post(f"{self.base_url}/activation/new", json=data)
        response.raise_for_status()
        return response.json()

    async def get_feature(self, model_id: str, layer: int, index: int) -> dict:
        response = await self.client.get(f"{self.base_url}/feature/{model_id}/{layer}/{index}")
        response.raise_for_status()
        return response.json()

    async def search_all_features(self, text: str, model_id: str = "gemma-2-2b", topk: int = 10) -> dict:
        data = {"text": text, "modelId": model_id, "topk": topk}
        response = await self.client.post(f"{self.base_url}/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_topk_by_token(self, text: str, model_id: str = "gemma-2-2b", topk: int = 10) -> dict:
        data = {"text": text, "modelId": model_id, "topk": topk}
        response = await self.client.post(f"{self.base_url}/search-topk-by-token", json=data)
        response.raise_for_status()
        return response.json()

    async def generate_explanation(self, model_id: str, layer: int, index: int) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/explanation/generate", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations(self, query: str, model_id: Optional[str] = None) -> dict:
        data = {"query": query}
        if model_id:
            data["modelId"] = model_id
        response = await self.client.post(f"{self.base_url}/explanation/search", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_generation(self, text: str, model_id: str, layer: int, index: int, multiplier: float = 1.0) -> dict:
        data = {
            "text": text,
            "modelId": model_id,
            "layer": layer,
            "index": index,
            "multiplier": multiplier
        }
        response = await self.client.post(f"{self.base_url}/steer", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_chat(self, messages: list, model_id: str, layer: int, index: int, multiplier: float = 1.0) -> dict:
        data = {
            "messages": messages,
            "modelId": model_id,
            "layer": layer,
            "index": index,
            "multiplier": multiplier
        }
        response = await self.client.post(f"{self.base_url}/steer-chat", json=data)
        response.raise_for_status()
        return response.json()

    async def list_graphs(self) -> dict:
        response = await self.client.get(f"{self.base_url}/graph/list")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


# Initialize FastMCP server
mcp = FastMCP("neuronpedia-mcp")

neuronpedia_client: Optional[NeuronpediaClient] = None


def get_client() -> NeuronpediaClient:
    global neuronpedia_client
    if neuronpedia_client is None:
        api_key = os.getenv("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("NEURONPEDIA_API_KEY environment variable is required")
        neuronpedia_client = NeuronpediaClient(api_key)
    return neuronpedia_client


@mcp.tool()
async def generate_attribution_graph(
    prompt: str,
    model_id: str = "gemma-2-2b",
    max_logits: Optional[int] = None,
    logit_probability: Optional[float] = None,
    node_threshold: Optional[float] = None,
    edge_threshold: Optional[float] = None,
) -> str:
    """Generate an attribution graph for analyzing text prompts.
    
    Args:
        prompt: Text prompt to analyze
        model_id: Model to use (default: gemma-2-2b)
        max_logits: Maximum number of logits (optional)
        logit_probability: Logit probability threshold (optional)
        node_threshold: Node threshold for graph (optional)
        edge_threshold: Edge threshold for graph (optional)
    """
    try:
        client = get_client()
        result = await client.generate_attribution_graph(
            prompt=prompt,
            model_id=model_id,
            max_logits=max_logits,
            logit_probability=logit_probability,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        
        return (f"Attribution Graph Generated Successfully!\n\n"
                f"🔗 **View Graph**: {result.url}\n"
                f"📊 **Nodes**: {result.numNodes}\n"
                f"🔗 **Links**: {result.numLinks}\n"
                f"💾 **Data**: {result.s3url}\n\n"
                f"The graph shows how different parts of the {model_id} model contribute "
                f"to generating each token in your prompt: '{prompt}'")
        
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_feature_activations(
    text: str,
    model_id: str,
    layer: int,
    index: int,
) -> str:
    """Get activation values for a specific feature on given text.
    
    Args:
        text: Input text to analyze
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: Layer number
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.get_activations(model_id, layer, index, text)
        return f"Feature Activations for {model_id} Layer {layer} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_feature_details(
    model_id: str,
    layer: int,
    index: int,
) -> str:
    """Get detailed information about a specific feature.
    
    Args:
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: Layer number
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.get_feature(model_id, layer, index)
        return f"Feature Details for {model_id} Layer {layer} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def search_top_features(
    text: str,
    model_id: str = "gemma-2-2b",
    topk: int = 10,
) -> str:
    """Find the top activating features for given text across the entire model.
    
    Args:
        text: Input text to analyze
        model_id: Model to search (default: gemma-2-2b)
        topk: Number of top features to return (default: 10)
    """
    try:
        client = get_client()
        result = await client.search_all_features(text, model_id, topk)
        return f"Top {topk} Features for '{text}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def search_features_by_token(
    text: str,
    model_id: str = "gemma-2-2b",
    topk: int = 10,
) -> str:
    """Find top activating features for each token in the text.
    
    Args:
        text: Input text to analyze
        model_id: Model to search (default: gemma-2-2b)
        topk: Number of top features per token (default: 10)
    """
    try:
        client = get_client()
        result = await client.search_topk_by_token(text, model_id, topk)
        return f"Top Features by Token for '{text}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def generate_feature_explanation(
    model_id: str,
    layer: int,
    index: int,
) -> str:
    """Generate an explanation for what a specific feature detects.
    
    Args:
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: Layer number
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.generate_explanation(model_id, layer, index)
        return f"Explanation for {model_id} Layer {layer} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def search_feature_explanations(
    query: str,
    model_id: Optional[str] = None,
) -> str:
    """Search for feature explanations across models.
    
    Args:
        query: Search query for explanations
        model_id: Optional model filter
    """
    try:
        client = get_client()
        result = await client.search_explanations(query, model_id)
        return f"Feature Explanations for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def steer_text_generation(
    text: str,
    model_id: str,
    layer: int,
    index: int,
    multiplier: float = 1.0,
) -> str:
    """Steer model text generation using a specific feature.
    
    Args:
        text: Input text to steer
        model_id: Model identifier
        layer: Layer number
        index: Feature index
        multiplier: Steering strength multiplier (default: 1.0)
    """
    try:
        client = get_client()
        result = await client.steer_generation(text, model_id, layer, index, multiplier)
        return f"Steered Generation Result:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def steer_chat_generation(
    messages: str,
    model_id: str,
    layer: int,
    index: int,
    multiplier: float = 1.0,
) -> str:
    """Steer chat model generation using a specific feature.
    
    Args:
        messages: JSON string of chat messages
        model_id: Model identifier
        layer: Layer number
        index: Feature index
        multiplier: Steering strength multiplier (default: 1.0)
    """
    try:
        import json
        client = get_client()
        messages_list = json.loads(messages)
        result = await client.steer_chat(messages_list, model_id, layer, index, multiplier)
        return f"Steered Chat Generation Result:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def list_user_graphs() -> str:
    """List all attribution graphs created by the user.
    """
    try:
        client = get_client()
        result = await client.list_graphs()
        return f"Your Attribution Graphs:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()