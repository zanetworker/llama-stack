# RedHat AI Validated Models - MCP Server

MCP (Model Context Protocol) server for searching and recommending RedHat AI validated models from HuggingFace using Llama Stack with **Contextual Retrieval**.
## Features

- 🔍 **Semantic Search** - Find models using natural language queries
- 🤖 **Smart Recommendations** - Get model suggestions based on use cases and constraints
- 📊 **Contextual Retrieval** - 35-67% better accuracy using Anthropic's contextual embeddings technique
- 🚀 **MCP Protocol** - Standard interface for AI tool integration
- 💾 **Vector Store** - FAISS-backed semantic search with nomic-embed-text (768d)

## Quick Start

### 1. Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Llama Stack Server

```bash
cd /path/to/llama-stack
source .venv/bin/activate
python -m llama_stack.cli.llama stack run experiments/ollama-setup/ollama-stack-run.yaml --port 8321
```

### 3. Ingest Model Cards (with Contextual Retrieval)

```bash
# Create vector store
python << 'EOF'
from llama_stack_client import LlamaStackClient
client = LlamaStackClient(base_url="http://localhost:8321")
vs = client.vector_stores.create(
    name="redhat_models_production",
    metadata={"purpose": "contextual_retrieval"},
    extra_body={
        "embedding_model": "ollama/nomic-embed-text:latest",
        "embedding_dimension": 768,
        "provider_id": "faiss"
    }
)
print(f"Vector Store ID: {vs.id}")
EOF

# Ingest models with contextual retrieval
python scripts/ingest_with_contextual_retrieval.py \
    --vector-store-id <vs_id> \
    --max-models 100

# Update .env with vector store ID
echo "VECTOR_DB_ID=<vs_id>" >> .env
```

### 4. Start MCP Server

```bash
# Start the MCP server
python -m src.mcp_server.server
```

### 5. Use MCP Tools

The server exposes these tools:
- `search_models` - Search for models using natural language
- `recommend_best_model` - Get recommendations based on use case
- `get_model_details` - Get full details for a specific model
- `list_all_models` - List all available models

## Architecture

```
┌─────────────────┐
│  MCP Client     │
│  (Claude, etc)  │
└────────┬────────┘
         │ MCP Protocol
┌────────▼────────┐
│  MCP Server     │
│  (This Project) │
└────────┬────────┘
         │
┌────────▼────────┐
│  Llama Stack    │
│  Vector Store   │
└────────┬────────┘
         │
┌────────▼────────┐
│  FAISS + Nomic  │
│  Embeddings     │
└─────────────────┘
```

## Contextual Retrieval

This project implements [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) technique:

1. **Manual Chunking** - Split documents into 700-token chunks
2. **Context Generation** - Use OpenAI to generate context for each chunk
3. **Context Prepending** - Add context before embedding
4. **Improved Matching** - 35-67% better retrieval accuracy

**Cost**: ~$0.006 per model (~$0.60 for 100 models)

See [TEST_RESULTS_CONTEXTUAL_RETRIEVAL.md](TEST_RESULTS_CONTEXTUAL_RETRIEVAL.md) for detailed results.

## Project Structure

```
.
├── src/
│   ├── config/          # Configuration and settings
│   ├── data/            # Data models and HuggingFace ingestion
│   ├── llama_stack/     # Llama Stack client wrapper
│   ├── mcp_server/      # MCP server implementation
│   ├── utils/           # Chunking and context generation
│   └── vector_db/       # Vector database utilities
├── scripts/
│   ├── ingest_with_contextual_retrieval.py  # Contextual ingestion
│   └── ingest_with_vector_stores_api.py     # Standard ingestion
└── tests/               # Test suite
```

## Configuration

Key environment variables in `.env`:

```bash
# Llama Stack
LLAMA_STACK_URL=http://localhost:8321
VECTOR_DB_ID=<your_vector_store_id>

# HuggingFace
HF_ORGANIZATION=RedHatAI
HF_TOKEN=<optional>

# OpenAI (for context generation)
OPENAI_API_KEY=<your_key>

# Embedding Model
VDB_EMBEDDING=nomic-embed-text
VDB_EMBEDDING_DIMENSION=768
VECTOR_DB_CHUNK_SIZE=800
```

## License

See [LICENSE](LICENSE) file.
