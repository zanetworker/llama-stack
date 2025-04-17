#!/usr/bin/env python3
"""
RAG Parameter Optimizer Example

This script demonstrates how to use the RAG Parameter Optimizer programmatically.
It shows how to get optimized parameters for different use cases and how to
interpret them for practical use in a RAG pipeline.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation
)

def print_parameters(params: Dict[str, Any], title: str) -> None:
    """
    Print parameters in a formatted way.
    
    Args:
        params: Dictionary of parameters
        title: Title to display
    """
    print(f"\n{'=' * 80}")
    print(f"= {title}")
    print(f"{'=' * 80}")
    
    print("\nRAG Parameters:")
    print(json.dumps(params, indent=2))
    
    print("\nPractical Interpretation:")
    chunk_size = parse_chunk_size(params.get('chunk_size_tokens', '~1000'))
    overlap = parse_overlap(params.get('overlap_tokens', '~100'))
    top_k = parse_top_k(params.get('top_k', '5'))
    embedding_model = map_embedding_recommendation(params.get('embedding_model_recommendation', ''))
    reranker = map_reranker_recommendation(params.get('retrieval_enhancements', ''))
    temperature = parse_temperature(params.get('generation_settings', ''))
    
    print(f"- Chunk Size: {chunk_size} tokens")
    print(f"- Chunk Overlap: {overlap} tokens")
    print(f"- Top-K: {top_k} chunks")
    print(f"- Embedding Model: {embedding_model}")
    print(f"- Reranker: {'Yes - ' + reranker if reranker else 'No'}")
    print(f"- Temperature: {temperature}")

def main():
    """Main function to demonstrate the RAG Parameter Optimizer."""
    # Example 1: Knowledge Management with technical documents
    params1 = get_rag_parameters(
        use_case="Knowledge Management",
        document_type="technical",
        performance_priority="accuracy",
        data_size="large"
    )
    print_parameters(params1, "Knowledge Management - Technical Documents - Accuracy Priority - Large Data")
    
    # Example 2: Customer Support with general documents
    params2 = get_rag_parameters(
        use_case="Customer Support",
        document_type="general",
        performance_priority="latency",
        data_size="medium"
    )
    print_parameters(params2, "Customer Support - General Documents - Latency Priority - Medium Data")
    
    # Example 3: Code Assistance with code documents
    params3 = get_rag_parameters(
        use_case="Code Assistance",
        document_type="code",
        performance_priority="balanced",
        data_size="small"
    )
    print_parameters(params3, "Code Assistance - Code Documents - Balanced Priority - Small Data")
    
    # Example 4: Healthcare Applications with technical documents
    params4 = get_rag_parameters(
        use_case="Healthcare Applications",
        document_type="technical",
        performance_priority="accuracy",
        data_size="medium"
    )
    print_parameters(params4, "Healthcare Applications - Technical Documents - Accuracy Priority - Medium Data")
    
    # Example 5: Custom use case (will use base defaults)
    params5 = get_rag_parameters(
        use_case="Financial Analysis",  # Not in the predefined use cases
        document_type="legal",
        performance_priority="cost",
        data_size="large"
    )
    print_parameters(params5, "Financial Analysis (Custom) - Legal Documents - Cost Priority - Large Data")

if __name__ == "__main__":
    main()
