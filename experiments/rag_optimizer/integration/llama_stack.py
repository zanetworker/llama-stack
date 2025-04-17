#!/usr/bin/env python3
"""
Llama Stack Integration for RAG Parameter Optimizer.

This script demonstrates how to use the RAG Parameter Optimizer with Llama Stack.
"""

import os
import sys
import json
from termcolor import colored

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation
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
            "model_name": map_llm_recommendation(parameters.get('generation_settings', '')),
            "temperature": parse_temperature(parameters.get('generation_settings', ''))
        }
    }

def main():
    """Main function to demonstrate the RAG parameter optimizer with Llama Stack."""
    # Get the model ID and port from environment variables
    model_id = os.environ.get("INFERENCE_MODEL", "llama3.2:3b-instruct-fp16")
    port = os.environ.get("LLAMA_STACK_PORT", "5002")
    
    print(colored(f"Using model: {model_id}", "green"))
    print(colored(f"Using Llama Stack port: {port}", "green"))
    
    # Example use cases to test
    test_cases = [
        {
            "name": "Knowledge Management with Technical Documents",
            "use_case": "Knowledge Management",
            "document_type": "technical",
            "performance_priority": "accuracy",
            "data_size": "large"
        },
        {
            "name": "Customer Support with Fast Response",
            "use_case": "Customer Support",
            "document_type": "general",
            "performance_priority": "latency",
            "data_size": "small"
        },
        {
            "name": "Code Assistance",
            "use_case": "Code Assistance",
            "document_type": "code",
            "performance_priority": "balanced",
            "data_size": "medium"
        },
        {
            "name": "Healthcare Applications",
            "use_case": "Healthcare Applications",
            "document_type": "technical",
            "performance_priority": "accuracy",
            "data_size": "large"
        }
    ]
    
    # Run tests
    for case in test_cases:
        print(colored(f"\n--- Test Case: {case['name']} ---", "cyan"))
        print(f"Use Case: {case['use_case']}")
        print(f"Document Type: {case['document_type']}")
        print(f"Performance Priority: {case['performance_priority']}")
        print(f"Data Size: {case['data_size']}")
        
        # Get optimized parameters
        parameters = get_rag_parameters(
            case['use_case'],
            case['document_type'],
            case['performance_priority'],
            case['data_size']
        )
        
        # Print parameters
        print(colored("\nOptimized Parameters:", "yellow"))
        print(json.dumps(parameters, indent=2))
        
        # Configure pipeline
        pipeline_config = configure_rag_pipeline(parameters)
        
        # Print pipeline configuration
        print(colored("\nPipeline Configuration:", "yellow"))
        print(json.dumps(pipeline_config, indent=2))
        
        print("-" * 80)

if __name__ == "__main__":
    main()
