# RAG Parameter Optimizer

A tool for optimizing Retrieval-Augmented Generation (RAG) parameters based on specific use cases and requirements.

## Overview

The RAG Parameter Optimizer helps you configure optimal RAG settings based on:

- **Use case** (question answering, summarization, document search, etc.)
- **Document type** (technical, legal, educational, code, etc.)
- **Performance priority** (accuracy, latency, cost, balanced)
- **Data size** (small, medium, large)

## Features

- **Parameter Optimization**: Get tailored RAG parameters for your specific use case
- **Multiple Interfaces**: CLI, Streamlit app, and programmatic API
- **Llama Stack Integration**: Seamlessly integrate with Llama Stack's RAG tools
- **Practical Interpretation**: Convert high-level recommendations to concrete parameter values

## Directory Structure

```
rag_optimizer/
├── app/                  # Streamlit application
│   ├── streamlit_app.py
│   └── streamlit_app_final.py
├── cli/                  # Command-line interface
│   └── rag_optimizer_cli.py
├── core/                 # Core optimization logic
│   ├── base_optimizer.py
│   └── enhanced_optimizer.py
├── integration/          # Integration with other systems
│   ├── llama_stack_integration.py
│   └── pipeline_example.py
└── tests/                # Test cases
    └── optimizer_test.py
```

## Usage

### CLI Interface

The CLI interface provides a simple way to get optimized RAG parameters from the command line.

```bash
# Basic usage
python rag_optimizer/cli/rag_optimizer_cli.py --use-case "Knowledge Management" --document-type "technical" --performance "accuracy" --data-size "large"

# Interactive mode
python rag_optimizer/cli/rag_optimizer_cli.py --interactive

# Save parameters to a file
python rag_optimizer/cli/rag_optimizer_cli.py --use-case "Code Assistance" --document-type "code" --output "code_rag_params.json"

# Different output formats
python rag_optimizer/cli/rag_optimizer_cli.py --use-case "Customer Support" --format "json"
python rag_optimizer/cli/rag_optimizer_cli.py --use-case "Customer Support" --format "compact"
```

### Streamlit App

The Streamlit app provides a user-friendly interface for exploring and visualizing RAG parameters.

```bash
# Run the Streamlit app
streamlit run rag_optimizer/app/streamlit_app_clean.py
```

### Programmatic API

You can also use the RAG Parameter Optimizer programmatically in your Python code.

```python
from rag_optimizer.core.enhanced_optimizer import get_rag_parameters

# Get optimized parameters
params = get_rag_parameters(
    use_case="Knowledge Management",
    document_type="technical",
    performance_priority="accuracy",
    data_size="large"
)

# Use the parameters in your RAG pipeline
print(params)
```

## Supported Use Cases

The RAG Parameter Optimizer supports the following use cases:

- **Knowledge Management**: Enterprise search, document retrieval, knowledge base Q&A
- **Customer Support**: FAQ automation, ticket resolution, support documentation search
- **Healthcare Applications**: Medical information retrieval, clinical decision support
- **Education**: Learning materials search, curriculum Q&A, study aids
- **Code Assistance**: Code search, documentation lookup, programming help
- **Sales Automation**: Product information retrieval, lead qualification
- **Marketing**: Content search, campaign analysis, trend summarization
- **Threat Analysis**: Security information retrieval, threat intelligence
- **Gaming**: Game lore, dynamic storytelling, player assistance

## Integration with Llama Stack

The RAG Parameter Optimizer can be integrated with Llama Stack's RAG tools and VectorIO APIs.

```python
from rag_optimizer.integration.llama_stack_integration import LlamaStackRAGIntegration

# Initialize integration
integration = LlamaStackRAGIntegration()

# Run RAG pipeline with optimized parameters
result = integration.run_rag_pipeline(
    use_case="Code Assistance",
    documents=documents,  # List of Document objects
    query="How do I implement a quick sort in Python?",
    document_type="code",
    performance_priority="balanced",
    data_size="small"
)

# Use the results
print(result["response"])
```

## Parameter Explanations

- **chunk_size_tokens**: The size of each text chunk in tokens
- **overlap_tokens**: The number of overlapping tokens between adjacent chunks
- **embedding_model_recommendation**: The recommended embedding model
- **top_k**: The number of chunks to retrieve for each query
- **metadata_usage**: How to use document metadata in retrieval
- **retrieval_enhancements**: Additional techniques to improve retrieval quality
- **generation_settings**: Settings for the language model that generates the final response
- **prompting_technique**: Recommended prompting strategy for the language model
- **output_guardrails**: Safeguards to ensure the quality and safety of generated responses
- **chunking_strategy_recommendation**: The recommended approach for dividing documents into chunks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
