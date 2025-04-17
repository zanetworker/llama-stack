# RAG Parameter Optimizer

This application helps optimize Retrieval-Augmented Generation (RAG) parameters based on specific use cases and requirements. It provides intelligent recommendations for RAG pipeline configuration to improve performance for different scenarios.

## Overview

The RAG Parameter Optimizer consists of several components:

1. **Parameter Optimizer Core**: Provides optimized RAG parameters based on use case, document type, performance priority, and data size.
2. **Streamlit Web Application**: A user-friendly interface for interacting with the optimizer, featuring both form-based and chat-based interfaces.
3. **Llama Stack Integration**: Tools for integrating the optimizer with the Llama Stack framework.

## Features

- **Use Case Optimization**: Tailored parameters for different use cases (Knowledge Management, Customer Support, Healthcare, Education, Code Assistance, etc.)
- **Document Type Adjustments**: Parameter adjustments based on document type (technical, legal, educational, code)
- **Performance Priority Tuning**: Optimization for different priorities (accuracy, latency, cost, balanced)
- **Data Size Scaling**: Parameter scaling based on data corpus size (small, medium, large)
- **Interactive Visualization**: Visual representation of the RAG pipeline and parameter impact
- **Parameter Customization**: Tools for fine-tuning recommended parameters
- **Chat Interface**: Natural language interaction for parameter recommendations

## Components

### 1. Parameter Optimizer Core

The core optimizer is implemented in two files:

- `rag_parameter_optimizer.py`: Base parameter recommendations for different use cases
- `rag_parameter_optimizer_enhanced.py`: Enhanced optimizer with additional adjustments for document type, performance priority, and data size

### 2. Streamlit Web Application

The Streamlit application provides a user-friendly interface for interacting with the optimizer:

- `rag_optimizer_app.py`: Streamlit application with form and chat interfaces

### 3. Llama Stack Integration

The Llama Stack integration provides tools for using the optimizer within the Llama Stack framework:

- `rag_optimizer_llama_integration.py`: Integration with Llama Stack as client tools

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Llama Stack Client (for Llama Stack integration)

### Installation

1. Install the required packages:

```bash
pip install streamlit pandas plotly streamlit-mermaid termcolor
```

2. For Llama Stack integration, install the Llama Stack Client:

```bash
pip install llama-stack-client
```

### Running the Streamlit Application

To run the Streamlit application:

```bash
cd /path/to/llama-stack
streamlit run experiments/rag_optimizer_app.py
```

This will launch the application in your default web browser.

### Using the Llama Stack Integration

To use the RAG Parameter Optimizer with Llama Stack:

1. Ensure Llama Stack is running
2. Set the required environment variables:

```bash
export LLAMA_STACK_PORT=8080
export INFERENCE_MODEL=meta-llama/Llama-3-8b-instruct
```

3. Run the integration script:

```bash
cd /path/to/llama-stack
python experiments/rag_optimizer_llama_integration.py
```

## Usage Examples

### Form Interface

1. Select your use case from the dropdown
2. Choose document type, performance priority, and data size
3. Click "Optimize Parameters" to get recommendations
4. View the recommended parameters and pipeline configuration
5. Customize parameters if needed

### Chat Interface

1. Describe your RAG use case in natural language
2. Include details about document type, performance priorities, and data size
3. Receive parameter recommendations and explanations
4. Ask for clarification or adjustments as needed

### Llama Stack Integration

```python
from llama_stack_client import LlamaStackClient
from experiments.rag_optimizer_llama_integration import optimize_rag_parameters, configure_rag_pipeline

# Get optimized parameters
params = optimize_rag_parameters(
    use_case="Knowledge Management",
    document_type="technical",
    performance_priority="accuracy",
    data_size="large"
)

# Configure RAG pipeline
pipeline_config = configure_rag_pipeline(params)

# Use the configuration in your RAG pipeline
# ...
```

## Parameter Explanations

- **chunk_size_tokens**: The size of each text chunk in tokens. Larger chunks provide more context but may reduce retrieval precision.
- **overlap_tokens**: The number of overlapping tokens between adjacent chunks. Higher overlap helps maintain context across chunk boundaries.
- **embedding_model_recommendation**: The recommended embedding model for converting text to vector representations.
- **top_k**: The number of chunks to retrieve for each query. Higher values provide more context but may introduce noise.
- **retrieval_enhancements**: Additional techniques to improve retrieval quality beyond basic vector search.
- **generation_settings**: Settings for the language model that generates the final response.
- **prompting_technique**: Recommended prompting strategy for the language model.
- **output_guardrails**: Safeguards to ensure the quality and safety of generated responses.
- **chunking_strategy_recommendation**: The recommended approach for dividing documents into chunks.

## Contributing

Contributions to the RAG Parameter Optimizer are welcome! Here are some ways to contribute:

- Add support for additional use cases
- Improve parameter recommendations
- Enhance the Streamlit application
- Add new integration options

## License

This project is licensed under the same license as the Llama Stack project.
