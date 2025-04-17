#!/usr/bin/env python3
"""
RAG Parameter Optimizer - Streamlit App

A clean, simple Streamlit interface for the RAG Parameter Optimizer.
Focuses on the form interface for getting optimized RAG parameters.
"""

import streamlit as st
import sys
import os
import json
import plotly.graph_objects as go

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the enhanced RAG parameter optimizer
from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature
)

def configure_rag_pipeline(parameters):
    """
    Configure a RAG pipeline based on the provided parameters.
    
    Args:
        parameters: Dictionary of RAG parameters
        
    Returns:
        Dictionary containing the configured pipeline components
    """
    # Parse parameters
    chunk_size = parse_chunk_size(parameters.get('chunk_size_tokens', '~1000'))
    overlap = parse_overlap(parameters.get('overlap_tokens', '~100'))
    chunk_strategy = parameters.get('chunking_strategy_recommendation', 'Recursive/Semantic').split('/')[0].lower()
    embed_model_name = map_embedding_recommendation(parameters.get('embedding_model_recommendation', ''))
    top_k = parse_top_k(parameters.get('top_k', '5'))
    retrieval_enhancements = parameters.get('retrieval_enhancements', '')
    
    # Return configuration details
    return {
        "text_splitter": {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "strategy": chunk_strategy
        },
        "embedding_model": {
            "model_name": embed_model_name
        },
        "retriever": {
            "top_k": top_k,
            "search_type": 'hybrid' if 'hybrid' in retrieval_enhancements.lower() else 'similarity'
        },
        "reranker": map_reranker_recommendation(retrieval_enhancements),
        "llm": {
            "temperature": parse_temperature(parameters.get('generation_settings', ''))
        }
    }

def generate_mermaid_diagram(pipeline_config):
    """
    Generate a Mermaid diagram for the RAG pipeline.
    
    Args:
        pipeline_config: Dictionary containing pipeline configuration
        
    Returns:
        String containing Mermaid diagram code
    """
    # Create Mermaid diagram
    mermaid_code = """
    graph LR
        A[Documents] --> B[Text Splitter]
        B --> C[Chunks]
        C --> D[Embedding Model]
        D --> E[Vector Store]
        E --> F[Retriever]
        F --> G[Retrieved Chunks]
    """
    
    # Add reranker if available
    if pipeline_config["reranker"]:
        mermaid_code += """
        G --> H[Reranker]
        H --> I[Ranked Chunks]
        I --> J[LLM]
        """
    else:
        mermaid_code += """
        G --> J[LLM]
        """
    
    mermaid_code += """
        K[Query] --> F
        K --> J
        J --> L[Response]
        
        classDef config fill:#f9f,stroke:#333,stroke-width:2px;
        class B,D,F,H,J config;
    """
    
    return mermaid_code

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(layout="wide", page_title="RAG Parameter Optimizer")
    
    st.title("RAG Parameter Optimizer")
    st.write("""
    This application helps you optimize your RAG (Retrieval-Augmented Generation) parameters 
    based on your specific use case and requirements.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Use case selection
        use_cases = [
            "Knowledge Management", 
            "Customer Support", 
            "Healthcare Applications", 
            "Education", 
            "Code Assistance", 
            "Sales Automation", 
            "Marketing", 
            "Threat Analysis", 
            "Gaming"
        ]
        use_case = st.selectbox("Select Use Case", use_cases)
        
        # Document type selection
        document_types = ["general", "technical", "legal", "educational", "code"]
        document_type = st.selectbox("Document Type", document_types)
        
        # Performance priority selection
        performance_priorities = ["balanced", "accuracy", "latency", "cost"]
        performance_priority = st.selectbox("Performance Priority", performance_priorities)
        
        # Data size selection
        data_sizes = ["small", "medium", "large"]
        data_size = st.selectbox("Data Size", data_sizes)
        
        # Button to optimize parameters
        optimize_button = st.button("Optimize Parameters")
    
    # Main content area
    if optimize_button:
        # Get optimized parameters
        parameters = get_rag_parameters(use_case, document_type, performance_priority, data_size)
        
        # Display parameters
        st.header("Optimized RAG Parameters")
        
        # Display parameters as JSON
        st.json(parameters)
        
        # Configure pipeline
        pipeline_config = configure_rag_pipeline(parameters)
        
        # Display pipeline configuration
        st.header("RAG Pipeline Configuration")
        
        # Create columns for different components
        col1, col2 = st.columns(2)
        
        with col1:
            # Display text splitter configuration
            st.subheader("Text Splitter")
            st.json(pipeline_config["text_splitter"])
            
            # Display embedding model configuration
            st.subheader("Embedding Model")
            st.json(pipeline_config["embedding_model"])
        
        with col2:
            # Display retriever configuration
            st.subheader("Retriever")
            st.json(pipeline_config["retriever"])
            
            # Display reranker configuration if available
            if pipeline_config["reranker"]:
                st.subheader("Reranker")
                st.write(f"Using reranker: {pipeline_config['reranker']}")
            
            # Display LLM configuration
            st.subheader("LLM")
            st.json(pipeline_config["llm"])
        
        # Display pipeline visualization
        st.header("RAG Pipeline Visualization")
        
        # Generate Mermaid diagram
        mermaid_code = generate_mermaid_diagram(pipeline_config)
        
        # Display Mermaid diagram
        try:
            # Try to import streamlit-mermaid
            from streamlit_mermaid import st_mermaid
            st_mermaid(mermaid_code)
        except ImportError:
            st.code(mermaid_code, language="mermaid")
            st.info("Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
        
        # Display parameter explanations
        st.header("Parameter Explanations")
        
        explanations = {
            "chunk_size_tokens": "The size of each text chunk in tokens. Larger chunks provide more context but may reduce retrieval precision.",
            "overlap_tokens": "The number of overlapping tokens between adjacent chunks. Higher overlap helps maintain context across chunk boundaries.",
            "embedding_model_recommendation": "The recommended embedding model for converting text to vector representations.",
            "top_k": "The number of chunks to retrieve for each query. Higher values provide more context but may introduce noise.",
            "retrieval_enhancements": "Additional techniques to improve retrieval quality beyond basic vector search.",
            "generation_settings": "Settings for the language model that generates the final response.",
            "prompting_technique": "Recommended prompting strategy for the language model.",
            "output_guardrails": "Safeguards to ensure the quality and safety of generated responses.",
            "chunking_strategy_recommendation": "The recommended approach for dividing documents into chunks."
        }
        
        # Create columns for explanations
        col1, col2 = st.columns(2)
        
        # Split explanations between columns
        params_list = list(parameters.items())
        half_point = len(params_list) // 2
        
        for i, (param, value) in enumerate(params_list):
            if param in explanations and param not in ["adjustments_applied", "use_case"]:
                with col1 if i < half_point else col2:
                    st.subheader(param)
                    st.write(f"**Value:** {value}")
                    st.write(f"**Explanation:** {explanations[param]}")
                    
                    # Add adjustment explanations based on document type, performance priority, and data size
                    if param == "chunk_size_tokens" and document_type != "general":
                        st.write(f"**Adjustment for {document_type} documents:** Chunk size was adjusted to optimize for {document_type} content.")
                    
                    if param == "embedding_model_recommendation" and performance_priority != "balanced":
                        st.write(f"**Adjustment for {performance_priority} priority:** Embedding model was selected to prioritize {performance_priority}.")
                    
                    if param == "top_k" and data_size != "medium":
                        st.write(f"**Adjustment for {data_size} data size:** Top-k value was adjusted to account for {data_size} data corpus.")

if __name__ == "__main__":
    main()
