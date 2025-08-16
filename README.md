# Neuronpedia MCP Server (Python)

A Python-based Model Context Protocol (MCP) server for the Neuronpedia API, providing AI model interpretability and feature analysis tools.

## Features

- **Attribution Graphs** - Generate visual attribution graphs using Gemma 2-2b model
- **Feature Analysis** - Analyze feature activations and search top features
- **Explanations** - Generate and search explanations for AI model features  
- **Model Steering** - Control model generation using specific features
- **Vector Management** - Create and manage custom steering vectors

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/manncodes/neuronpedia-mcp-python.git
cd neuronpedia-mcp-python
uv sync
```

### Using pip

```bash
git clone https://github.com/manncodes/neuronpedia-mcp-python.git
cd neuronpedia-mcp-python
pip install -e .
```

## Configuration

Set your Neuronpedia API key:

```bash
export NEURONPEDIA_API_KEY=your_api_key_here
```

Get your API key from [neuronpedia.org](https://neuronpedia.org) account page.

## Usage

### Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neuronpedia": {
      "command": "uv",
      "args": ["run", "python", "src/neuronpedia_mcp/server.py"],
      "cwd": "/path/to/neuronpedia-mcp-python",
      "env": {
        "NEURONPEDIA_API_KEY": "your_api_key"
      }
    }
  }
}
```

### Direct Usage

```bash
export NEURONPEDIA_API_KEY=your_key
uv run python src/neuronpedia_mcp/server.py
```

## Available Tools

1. **generate_attribution_graph** - Generate attribution graphs for text analysis
2. **generate_explanation** - Generate explanations for model features
3. **search_explanations** - Search existing explanations
4. **get_activations** - Get feature activations for text
5. **search_top_features** - Find top activating features
6. **steer_generation** - Steer model generation with features

## Testing

Test the API connection:
```bash
uv run python test_mcp.py api
```

Test the full MCP server:
```bash
uv run python test_mcp.py
```

## Debugging

- Check logs in Claude Desktop: `%APPDATA%\Claude\logs\mcp-server-neuronpedia.log`
- Use the test script to verify API connectivity
- Ensure API key is correctly set

## License

MIT