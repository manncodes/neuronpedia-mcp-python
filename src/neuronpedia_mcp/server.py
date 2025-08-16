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

    async def delete_graph(self, model_id: str, slug: str) -> dict:
        data = {"modelId": model_id, "slug": slug}
        response = await self.client.post(f"{self.base_url}/graph/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def get_graph_metadata(self, model_id: str, slug: str) -> dict:
        response = await self.client.get(f"{self.base_url}/graph/{model_id}/{slug}")
        response.raise_for_status()
        return response.json()

    # Bookmarks
    async def add_bookmark(self, model_id: str, layer: str, index: str) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/bookmark/add", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_bookmark(self, model_id: str, layer: str, index: str) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/bookmark/delete", json=data)
        response.raise_for_status()
        return response.json()

    # Lists
    async def create_list(self, name: str, description: str = "", test_text: Optional[str] = None) -> dict:
        data = {"name": name, "description": description}
        if test_text:
            data["testText"] = test_text
        response = await self.client.post(f"{self.base_url}/list/new", json=data)
        response.raise_for_status()
        return response.json()

    async def create_list_with_features(self, name: str, features: list, description: str = "", test_text: Optional[str] = None) -> dict:
        data = {"name": name, "features": features, "description": description}
        if test_text:
            data["testText"] = test_text
        response = await self.client.post(f"{self.base_url}/list/new-with-features", json=data)
        response.raise_for_status()
        return response.json()

    async def get_user_lists(self) -> dict:
        response = await self.client.post(f"{self.base_url}/list/list")
        response.raise_for_status()
        return response.json()

    async def get_list_details(self, list_id: str) -> dict:
        data = {"listId": list_id}
        response = await self.client.post(f"{self.base_url}/list/get", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_list(self, list_id: str) -> dict:
        data = {"listId": list_id}
        response = await self.client.post(f"{self.base_url}/list/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def add_features_to_list(self, list_id: str, features: list) -> dict:
        data = {"listId": list_id, "featuresToAdd": features}
        response = await self.client.post(f"{self.base_url}/list/add-features", json=data)
        response.raise_for_status()
        return response.json()

    async def remove_feature_from_list(self, list_id: str, model_id: str, layer: str, index: str) -> dict:
        data = {"listId": list_id, "modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/list/remove", json=data)
        response.raise_for_status()
        return response.json()

    # Vectors
    async def create_vector(self, model_id: str, layer_number: int, vector: list, vector_label: str, default_steer_strength: float, hook_type: str = "resid-pre") -> dict:
        data = {
            "modelId": model_id,
            "layerNumber": layer_number,
            "hookType": hook_type,
            "vector": vector,
            "vectorDefaultSteerStrength": default_steer_strength,
            "vectorLabel": vector_label
        }
        response = await self.client.post(f"{self.base_url}/vector/new", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_vector(self, model_id: str, source: str, index: str) -> dict:
        data = {"modelId": model_id, "source": source, "index": index}
        response = await self.client.post(f"{self.base_url}/vector/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def get_vector_details(self, model_id: str, source: str, index: str) -> dict:
        data = {"modelId": model_id, "source": source, "index": index}
        response = await self.client.post(f"{self.base_url}/vector/get", json=data)
        response.raise_for_status()
        return response.json()

    async def list_user_vectors(self) -> dict:
        response = await self.client.post(f"{self.base_url}/vector/list-owned")
        response.raise_for_status()
        return response.json()

    # Enhanced Explanation methods
    async def score_explanation(self, explanation_id: str, scorer_model: str, scorer_type: str) -> dict:
        data = {"explanationId": explanation_id, "scorerModel": scorer_model, "scorerType": scorer_type}
        response = await self.client.post(f"{self.base_url}/explanation/score", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_explanation(self, explanation_id: str) -> dict:
        response = await self.client.post(f"{self.base_url}/explanation/{explanation_id}/delete")
        response.raise_for_status()
        return response.json()

    # Enhanced search methods
    async def search_explanations_all(self, query: str, offset: int = 0) -> dict:
        data = {"query": query, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations_by_model(self, query: str, model_id: str, offset: int = 0) -> dict:
        data = {"query": query, "modelId": model_id, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search-model", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations_by_layers(self, query: str, model_id: str, layers: list, offset: int = 0) -> dict:
        data = {"query": query, "modelId": model_id, "layers": layers, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search", json=data)
        response.raise_for_status()
        return response.json()

    # Model management
    async def create_model(self, model_id: str, layers: int, display_name: str = "", url: Optional[str] = None) -> dict:
        data = {"id": model_id, "layers": layers}
        if display_name:
            data["displayName"] = display_name
        if url:
            data["url"] = url
        response = await self.client.post(f"{self.base_url}/model/new", json=data)
        response.raise_for_status()
        return response.json()

    # Advanced search
    async def search_all_features_advanced(self, model_id: str, source_set: str, text: str, selected_layers: list = None, sort_indexes: list = None, num_results: int = 50, ignore_bos: bool = False, density_threshold: float = -1) -> dict:
        data = {
            "modelId": model_id,
            "sourceSet": source_set,
            "text": text,
            "selectedLayers": selected_layers or [],
            "sortIndexes": sort_indexes or [],
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold
        }
        response = await self.client.post(f"{self.base_url}/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_topk_by_token_advanced(self, model_id: str, source: str, text: str, num_results: int = 10, ignore_bos: bool = True, density_threshold: float = 0.01) -> dict:
        data = {
            "modelId": model_id,
            "source": source,
            "text": text,
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold
        }
        response = await self.client.post(f"{self.base_url}/search-topk-by-token", json=data)
        response.raise_for_status()
        return response.json()

    # Enhanced steering
    async def steer_chat_advanced(self, default_messages: list, steered_messages: list, model_id: str, features: list, temperature: float = 0.5, n_tokens: int = 48, freq_penalty: float = 2, seed: int = 16, strength_multiplier: float = 4, steer_special_tokens: bool = True) -> dict:
        data = {
            "defaultChatMessages": default_messages,
            "steeredChatMessages": steered_messages,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_special_tokens": steer_special_tokens
        }
        response = await self.client.post(f"{self.base_url}/steer-chat", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_text_advanced(self, prompt: str, model_id: str, features: list, temperature: float = 0.5, n_tokens: int = 48, freq_penalty: float = 2, seed: int = 16, strength_multiplier: float = 4) -> dict:
        data = {
            "prompt": prompt,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier
        }
        response = await self.client.post(f"{self.base_url}/steer", json=data)
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


# Graph Management
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


@mcp.tool()
async def delete_attribution_graph(model_id: str, slug: str) -> str:
    """Delete an attribution graph you created.
    
    Args:
        model_id: Model identifier 
        slug: Graph slug identifier
    """
    try:
        client = get_client()
        result = await client.delete_graph(model_id, slug)
        return f"Graph deleted successfully: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Bookmark Management
@mcp.tool()
async def add_feature_bookmark(model_id: str, layer: str, index: str) -> str:
    """Add a feature to your bookmarks.
    
    Args:
        model_id: Model identifier
        layer: Layer or SAE identifier
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.add_bookmark(model_id, layer, index)
        return f"Bookmark added for {model_id} layer {layer} feature {index}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def remove_feature_bookmark(model_id: str, layer: str, index: str) -> str:
    """Remove a feature from your bookmarks.
    
    Args:
        model_id: Model identifier
        layer: Layer or SAE identifier
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.delete_bookmark(model_id, layer, index)
        return f"Bookmark removed for {model_id} layer {layer} feature {index}"
    except Exception as e:
        return f"Error: {str(e)}"


# List Management
@mcp.tool()
async def create_feature_list(name: str, description: str = "", test_text: Optional[str] = None) -> str:
    """Create a new feature list.
    
    Args:
        name: Name of the list
        description: Optional description
        test_text: Optional test text for activation visualization
    """
    try:
        client = get_client()
        result = await client.create_list(name, description, test_text)
        return f"Feature list created: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_user_feature_lists() -> str:
    """Get all feature lists created by the user.
    """
    try:
        client = get_client()
        result = await client.get_user_lists()
        return f"Your Feature Lists:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def delete_feature_list(list_id: str) -> str:
    """Delete a feature list.
    
    Args:
        list_id: ID of the list to delete
    """
    try:
        client = get_client()
        result = await client.delete_list(list_id)
        return f"Feature list deleted: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Vector Management
@mcp.tool()
async def create_steering_vector(
    model_id: str,
    layer_number: int,
    vector_values: str,
    vector_label: str,
    default_steer_strength: float,
    hook_type: str = "resid-pre"
) -> str:
    """Create a new steering vector.
    
    Args:
        model_id: Model identifier
        layer_number: Layer number (0-based)
        vector_values: JSON array of vector values
        vector_label: Label for the vector
        default_steer_strength: Default steering strength (-100 to 100)
        hook_type: Hook type (default: resid-pre)
    """
    try:
        import json
        client = get_client()
        vector = json.loads(vector_values)
        result = await client.create_vector(model_id, layer_number, vector, vector_label, default_steer_strength, hook_type)
        return f"Steering vector created: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def list_user_vectors() -> str:
    """List all vectors created by the user.
    """
    try:
        client = get_client()
        result = await client.list_user_vectors()
        return f"Your Steering Vectors:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_vector_details(model_id: str, source: str, index: str) -> str:
    """Get details of a specific vector.
    
    Args:
        model_id: Model identifier
        source: Source identifier
        index: Vector index
    """
    try:
        client = get_client()
        result = await client.get_vector_details(model_id, source, index)
        return f"Vector Details:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Enhanced Search
@mcp.tool()
async def search_all_explanations(query: str, offset: int = 0) -> str:
    """Search explanations across all features on Neuronpedia.
    
    Args:
        query: Search query
        offset: Pagination offset (default: 0)
    """
    try:
        client = get_client()
        result = await client.search_explanations_all(query, offset)
        return f"Search Results for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def search_explanations_by_model_advanced(query: str, model_id: str, offset: int = 0) -> str:
    """Search explanations within a specific model.
    
    Args:
        query: Search query (minimum 3 characters)
        model_id: Model to search within
        offset: Pagination offset (default: 0)
    """
    try:
        client = get_client()
        result = await client.search_explanations_by_model(query, model_id, offset)
        return f"Search Results in {model_id} for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def advanced_feature_search(
    model_id: str,
    source_set: str,
    text: str,
    selected_layers: str = "[]",
    sort_indexes: str = "[]", 
    num_results: int = 50,
    ignore_bos: bool = False,
    density_threshold: float = -1
) -> str:
    """Advanced feature search with multiple parameters.
    
    Args:
        model_id: Model to search
        source_set: SAE set to search
        text: Text to analyze
        selected_layers: JSON array of layer IDs to search (default: all)
        sort_indexes: JSON array of token indexes to sort by (default: max activation)
        num_results: Max results to return (default: 50, max: 100)
        ignore_bos: Don't return results where top activation is BOS token
        density_threshold: Don't return features above this density (0-1, -1 = no threshold)
    """
    try:
        import json
        client = get_client()
        layers = json.loads(selected_layers)
        indexes = json.loads(sort_indexes)
        result = await client.search_all_features_advanced(
            model_id, source_set, text, layers, indexes, num_results, ignore_bos, density_threshold
        )
        return f"Advanced Search Results for '{text}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def score_feature_explanation(explanation_id: str, scorer_model: str, scorer_type: str) -> str:
    """Score an explanation using AI models.
    
    Args:
        explanation_id: ID of explanation to score
        scorer_model: Model to use for scoring (e.g., gpt-4o-mini, gemini-1.5-flash)
        scorer_type: Scoring method (recall_alt, eleuther_fuzz, eleuther_recall, eleuther_embedding)
    """
    try:
        client = get_client()
        result = await client.score_explanation(explanation_id, scorer_model, scorer_type)
        return f"Explanation Scoring Result:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def advanced_text_steering(
    prompt: str,
    model_id: str,
    features_json: str,
    temperature: float = 0.5,
    n_tokens: int = 48,
    freq_penalty: float = 2.0,
    seed: int = 16,
    strength_multiplier: float = 4.0
) -> str:
    """Advanced text steering with multiple features and parameters.
    
    Args:
        prompt: Text prompt to steer
        model_id: Model identifier
        features_json: JSON array of feature objects with modelId, layer, index, strength
        temperature: Generation temperature (default: 0.5)
        n_tokens: Number of tokens to generate (default: 48)
        freq_penalty: Frequency penalty (default: 2.0)
        seed: Random seed (default: 16)
        strength_multiplier: Global strength multiplier (default: 4.0)
    """
    try:
        import json
        client = get_client()
        features = json.loads(features_json)
        result = await client.steer_text_advanced(
            prompt, model_id, features, temperature, n_tokens, freq_penalty, seed, strength_multiplier
        )
        return f"Advanced Steering Result:\n\n{result}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()