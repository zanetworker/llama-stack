# RAG Method Relationships and Dependencies

## Overview

This document provides a detailed analysis of how the 3 RAG query methods in Llama Stack relate to and depend on each other. Understanding these relationships is crucial for choosing the right method and debugging issues.

## Architectural Layers

```mermaid
graph TB
    subgraph "Application Layer"
        A1[User Application]
    end
    
    subgraph "Agent Layer"
        B1[Agent Framework]
        B2[builtin::rag toolgroup]
        B3[Agent Session Management]
    end
    
    subgraph "Tool Runtime Layer"
        C1[ToolRuntimeRouter]
        C2[MemoryToolRuntimeImpl]
        C3[Context Retriever]
        C4[Query Generation]
    end
    
    subgraph "Vector IO Layer"
        D1[VectorIO Protocol]
        D2[Provider Implementations]
    end
    
    subgraph "Database Layer"
        E1[Faiss]
        E2[Chroma]
        E3[Qdrant]
        E4[Milvus]
        E5[PGVector]
        E6[SQLite-vec]
    end
    
    A1 --> B1
    A1 --> C2
    A1 --> D1
    
    B1 --> B2
    B2 --> C1
    C1 --> C2
    
    C2 --> C3
    C2 --> D1
    C3 --> C4
    
    D1 --> D2
    D2 --> E1
    D2 --> E2
    D2 --> E3
    D2 --> E4
    D2 --> E5
    D2 --> E6
    
    style B1 fill:#e1f5fe
    style C2 fill:#fff3e0
    style D1 fill:#e8f5e8
```

## Method Dependencies

### 1. Agent-based RAG Dependencies

**Direct Dependencies:**
- [`ToolRuntimeRouter`](../../llama_stack/distribution/routers/tool_runtime.py#L28-L60)
- Agent framework and session management
- Event logging and streaming infrastructure

**Indirect Dependencies:**
- RAG Tool API (via ToolRuntimeRouter)
- Vector IO API (via RAG Tool API)
- Vector database providers (via Vector IO API)

**Code Flow:**
```python
# Agent configuration triggers this flow:
Agent.create_turn() 
  → AgentInstance._run_to_completion()
  → ToolRuntimeRouter.invoke_tool("knowledge_search")
  → MemoryToolRuntimeImpl.invoke_tool()
  → MemoryToolRuntimeImpl.query()
  → VectorIO.query_chunks()
```

### 2. RAG Tool API Dependencies

**Direct Dependencies:**
- [`VectorIO`](../../llama_stack/apis/vector_io/vector_io.py#L238-L275) protocol
- [`Inference`](../../llama_stack/apis/inference) API (for LLM query generation)
- Context retriever and query generation modules

**Implementation Location:**
- Core: [`MemoryToolRuntimeImpl`](../../llama_stack/providers/inline/tool_runtime/rag/memory.py#L52-L225)
- Query processing: [`generate_rag_query()`](../../llama_stack/providers/inline/tool_runtime/rag/context_retriever.py#L23-L38)

**Key Integration Points:**
```python
# RAG Tool API calls Vector IO API:
async def query(self, content, vector_db_ids, query_config):
    # Generate optimized query
    query = await generate_rag_query(query_config.query_generator_config, content)
    
    # Call Vector IO for each database
    tasks = [
        self.vector_io_api.query_chunks(
            vector_db_id=vector_db_id,
            query=query,
            params={
                "mode": query_config.mode,
                "max_chunks": query_config.max_chunks,
                "ranker": query_config.ranker,
            }
        )
        for vector_db_id in vector_db_ids
    ]
    results = await asyncio.gather(*tasks)
```

### 3. Vector IO API Dependencies

**Direct Dependencies:**
- Vector database provider implementations
- Embedding models (for query vectorization)
- Database-specific client libraries

**Provider Implementations:**
- [`FaissVectorIOAdapter`](../../llama_stack/providers/inline/vector_io/faiss/faiss.py)
- [`ChromaVectorIOAdapter`](../../llama_stack/providers/remote/vector_io/chroma/chroma.py)
- [`QdrantVectorIOAdapter`](../../llama_stack/providers/remote/vector_io/qdrant/qdrant.py)
- [`MilvusVectorIOAdapter`](../../llama_stack/providers/remote/vector_io/milvus/milvus.py)
- [`PGVectorVectorIOAdapter`](../../llama_stack/providers/remote/vector_io/pgvector/pgvector.py)
- [`SQLiteVecVectorIOAdapter`](../../llama_stack/providers/inline/vector_io/sqlite_vec/sqlite_vec.py)

## Data Flow Analysis

### Complete Query Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant ToolRouter
    participant RAGTool
    participant ContextRetriever
    participant VectorIO
    participant Provider
    participant VectorDB
    
    Note over User,VectorDB: Agent-based RAG Flow
    User->>Agent: "What is machine learning?"
    Agent->>Agent: Determine tool needed
    Agent->>ToolRouter: invoke_tool("knowledge_search", {"query": "..."})
    
    Note over ToolRouter,RAGTool: Tool Runtime Layer
    ToolRouter->>RAGTool: query(content, vector_db_ids, config)
    RAGTool->>ContextRetriever: generate_rag_query(config, content)
    ContextRetriever-->>RAGTool: optimized_query
    
    Note over RAGTool,VectorDB: Vector IO Layer
    RAGTool->>VectorIO: query_chunks(vector_db_id, query, params)
    VectorIO->>Provider: query_chunks(vector_db_id, query, params)
    Provider->>VectorDB: search(embedding, k, filters)
    VectorDB-->>Provider: raw_results
    Provider-->>VectorIO: QueryChunksResponse
    VectorIO-->>RAGTool: QueryChunksResponse
    
    Note over RAGTool,User: Context Assembly
    RAGTool->>RAGTool: format_context(chunks, template)
    RAGTool->>RAGTool: apply_token_limits()
    RAGTool-->>ToolRouter: RAGQueryResult
    ToolRouter-->>Agent: ToolInvocationResult
    Agent->>Agent: Generate response with context
    Agent-->>User: "Based on the retrieved information..."
```

### Error Propagation

```mermaid
graph TD
    A[Vector DB Error] --> B[Provider Error]
    B --> C[Vector IO Error]
    C --> D[RAG Tool Error]
    D --> E[Tool Router Error]
    E --> F[Agent Error]
    F --> G[User Error Response]
    
    style A fill:#ffebee
    style G fill:#ffebee
```

## Configuration Inheritance

### How Configuration Flows Through Layers

```mermaid
graph LR
    subgraph "Agent Config"
        A1[toolgroups.args]
        A2[vector_db_ids]
        A3[query_config]
    end
    
    subgraph "RAG Tool Config"
        B1[RAGQueryConfig]
        B2[max_chunks]
        B3[max_tokens_in_context]
        B4[mode]
        B5[ranker]
    end
    
    subgraph "Vector IO Params"
        C1[max_chunks]
        C2[mode]
        C3[score_threshold]
        C4[ranker]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B2 --> C1
    B4 --> C2
    B5 --> C4
    
    style A1 fill:#e1f5fe
    style B1 fill:#fff3e0
    style C1 fill:#e8f5e8
```

### Configuration Examples

**Agent Level:**
```python
agent_config = {
    "toolgroups": [{
        "name": "builtin::rag",
        "args": {
            "vector_db_ids": ["kb1", "kb2"],
            "query_config": {
                "max_chunks": 5,
                "max_tokens_in_context": 2048,
                "mode": "hybrid",
                "ranker": {"type": "rrf", "impact_factor": 60.0}
            }
        }
    }]
}
```

**RAG Tool Level:**
```python
rag_config = RAGQueryConfig(
    max_chunks=5,
    max_tokens_in_context=2048,
    mode="hybrid",
    ranker=RRFRanker(impact_factor=60.0),
    chunk_template="Result {index}\nContent: {chunk.content}\n"
)
```

**Vector IO Level:**
```python
vector_params = {
    "max_chunks": 5,
    "mode": "hybrid",
    "score_threshold": 0.0,
    "ranker": {"type": "rrf", "impact_factor": 60.0}
}
```

## Performance Implications

### Method Overhead Comparison

| Method | Overhead | Reason |
|--------|----------|---------|
| Agent-based RAG | High | Agent processing, session management, event logging |
| RAG Tool API | Medium | Query generation, context formatting, multi-DB aggregation |
| Vector IO API | Low | Direct database access, minimal processing |

### Optimization Strategies by Layer

**Agent Layer Optimizations:**
- Use session persistence to avoid re-initialization
- Configure appropriate `max_infer_iters` to limit agent loops
- Enable streaming for real-time responses

**RAG Tool Layer Optimizations:**
- Use `DefaultRAGQueryGeneratorConfig` instead of LLM-based generation for speed
- Optimize `chunk_template` to minimize token usage
- Set appropriate `max_tokens_in_context` limits

**Vector IO Layer Optimizations:**
- Choose appropriate vector database for your use case
- Optimize embedding dimensions and chunk sizes
- Use appropriate search modes (vector vs hybrid)

## Debugging Relationships

### Common Issues and Their Sources

**Issue: No results returned**
```
Agent → ToolRouter → RAGTool → VectorIO → Provider → VectorDB
                                                    ↑
                                            Check here first
```

**Issue: Poor quality results**
```
Agent → ToolRouter → RAGTool → ContextRetriever
                              ↑
                        Check query generation
```

**Issue: Performance problems**
```
Agent → ToolRouter → RAGTool → VectorIO
↑                            ↑
Agent overhead              Database performance
```

### Debugging Tools

**Enable Debug Logging:**
```python
import logging
logging.getLogger("llama_stack").setLevel(logging.DEBUG)
```

**Check Each Layer:**
```python
# Test Vector IO directly
response = client.vector_io.query_chunks(vector_db_id, query, params)

# Test RAG Tool directly  
result = client.tool_runtime.rag_tool.query(content, vector_db_ids, config)

# Test Agent with RAG
agent_response = agent.create_turn(messages, session_id)
```

## Integration Patterns

### Pattern 1: Progressive Enhancement
Start with Vector IO, add RAG Tool features, then wrap in Agent:

```python
# Step 1: Vector IO
chunks = client.vector_io.query_chunks(vector_db_id, query, params)

# Step 2: Add RAG Tool processing
result = client.tool_runtime.rag_tool.query(content, vector_db_ids, config)

# Step 3: Wrap in Agent
agent = Agent(client, {"toolgroups": [{"name": "builtin::rag", "args": {...}}]})
```

### Pattern 2: Hybrid Approach
Use different methods for different use cases:

```python
# Fast lookups: Vector IO
if query_type == "simple_lookup":
    return client.vector_io.query_chunks(vector_db_id, query, params)

# Complex queries: RAG Tool
elif query_type == "complex_analysis":
    return client.tool_runtime.rag_tool.query(content, vector_db_ids, config)

# Conversational: Agent
elif query_type == "conversation":
    return agent.create_turn(messages, session_id)
```

### Pattern 3: Fallback Chain
Try methods in order of sophistication:

```python
try:
    # Try agent first for full context
    return agent.create_turn(messages, session_id)
except AgentError:
    try:
        # Fallback to RAG Tool
        return client.tool_runtime.rag_tool.query(content, vector_db_ids, config)
    except RAGToolError:
        # Final fallback to Vector IO
        return client.vector_io.query_chunks(vector_db_id, query, params)
```

## Summary

The 3 RAG methods form a **layered architecture** where:

1. **Vector IO API** provides the foundation with direct database access
2. **RAG Tool API** builds on Vector IO, adding query processing and context formatting
3. **Agent-based RAG** builds on RAG Tool, adding conversational context and automation

Understanding these relationships helps you:
- Choose the right method for your use case
- Debug issues by checking the appropriate layer
- Optimize performance at the right level
- Migrate between methods as requirements change

Each layer adds value while maintaining compatibility with the layers below, providing a flexible and powerful RAG ecosystem.