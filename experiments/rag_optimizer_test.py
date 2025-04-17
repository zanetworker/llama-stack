#!/usr/bin/env python3
"""
Test script for the RAG Parameter Optimizer.

This script demonstrates how to use the RAG Parameter Optimizer directly from Python,
without the Streamlit interface or Llama Stack integration.
"""

import json
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced RAG parameter optimizer
from experiments.rag_parameter_optimizer_enhanced import get_rag_parameters
from experiments.rag_integration_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation,
    configure_and_run_rag
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

def test_optimizer():
    """Test the RAG Parameter Optimizer with various use cases and configurations."""
    print("=== RAG Parameter Optimizer Test ===\n")
    
    # Test cases
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
    for test_case in test_cases:
        print(f"\n--- Test Case: {test_case['name']} ---")
        print(f"Use Case: {test_case['use_case']}")
        print(f"Document Type: {test_case['document_type']}")
        print(f"Performance Priority: {test_case['performance_priority']}")
        print(f"Data Size: {test_case['data_size']}")
        
        # Get optimized parameters
        parameters = get_rag_parameters(
            test_case['use_case'],
            test_case['document_type'],
            test_case['performance_priority'],
            test_case['data_size']
        )
        
        # Print parameters
        print("\nOptimized Parameters:")
        print(json.dumps(parameters, indent=2))
        
        # Configure pipeline
        pipeline_config = configure_rag_pipeline(parameters)
        
        # Print pipeline configuration
        print("\nPipeline Configuration:")
        print(json.dumps(pipeline_config, indent=2))
        
        print("-" * 80)

def test_integration():
    """Test the integration with the RAG pipeline."""
    print("\n=== RAG Pipeline Integration Test ===\n")
    
    # Test with a code assistance use case
    print("Testing Code Assistance use case with a code snippet...")
    
    code_snippet = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Another function
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    
    # Configure and run RAG with optimized parameters
    configure_and_run_rag(
        use_case="Code Assistance", 
        sample_query="How do I implement a quick sort in Python?",
        sample_text=code_snippet
    )
    
    # Test with a customer support use case
    print("\nTesting Customer Support use case with a FAQ snippet...")
    
    faq_snippet = """
To reset your password, go to the login page and click 'Forgot Password'. 
Enter your email address and follow the instructions sent to you. 
If you don't receive an email, check your spam folder or contact support.

To update your billing information, go to Account Settings > Billing. 
You can update your credit card, billing address, and other payment details there.

For refunds, please contact our support team within 30 days of purchase. 
Include your order number and reason for the refund in your request.
"""
    
    # Configure and run RAG with optimized parameters
    configure_and_run_rag(
        use_case="Customer Support", 
        sample_query="How do I reset my password?",
        sample_text=faq_snippet
    )

if __name__ == "__main__":
    # Test the optimizer
    test_optimizer()
    
    # Test the integration
    test_integration()
