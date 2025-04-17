#!/usr/bin/env python3
"""
RAG Parameter Optimizer CLI

This script provides a command-line interface for the RAG Parameter Optimizer.
It allows users to get optimized RAG parameters based on their use case and requirements.

Usage:
    python rag_optimizer_cli.py --use-case "Knowledge Management" --document-type "technical" --performance "accuracy" --data-size "large"
    python rag_optimizer_cli.py --interactive
    python rag_optimizer_cli.py --help
"""

import argparse
import json
import sys
import os
from typing import Dict, Any, Optional

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation
)

# Available options for command-line arguments
USE_CASES = [
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

DOCUMENT_TYPES = ["general", "technical", "legal", "educational", "code"]
PERFORMANCE_PRIORITIES = ["balanced", "accuracy", "latency", "cost"]
DATA_SIZES = ["small", "medium", "large"]

def get_optimized_parameters(
    use_case: str,
    document_type: str = "general",
    performance_priority: str = "balanced",
    data_size: str = "medium"
) -> Dict[str, Any]:
    """
    Get optimized RAG parameters based on the specified criteria.
    
    Args:
        use_case: The primary use case (e.g., "Knowledge Management", "Customer Support")
        document_type: Type of documents (e.g., "technical", "legal", "educational", "code")
        performance_priority: Optimization priority (e.g., "accuracy", "latency", "cost", "balanced")
        data_size: Size of the data corpus (e.g., "small", "medium", "large")
        
    Returns:
        Dictionary of optimized RAG parameters
    """
    return get_rag_parameters(use_case, document_type, performance_priority, data_size)

def save_parameters_to_file(parameters: Dict[str, Any], output_file: str) -> None:
    """
    Save parameters to a JSON file.
    
    Args:
        parameters: Dictionary of parameters to save
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        json.dump(parameters, f, indent=2)
    print(f"Parameters saved to {output_file}")

def print_parameters(parameters: Dict[str, Any], format_type: str = "pretty") -> None:
    """
    Print parameters to the console.
    
    Args:
        parameters: Dictionary of parameters to print
        format_type: Format type ("pretty", "json", or "compact")
    """
    if format_type == "json":
        print(json.dumps(parameters))
    elif format_type == "compact":
        # Print key parameters in a compact format
        print(f"Use Case: {parameters.get('use_case', 'Not specified')}")
        print(f"Chunk Size: {parameters.get('chunk_size_tokens', 'Not specified')}")
        print(f"Overlap: {parameters.get('overlap_tokens', 'Not specified')}")
        print(f"Embedding Model: {parameters.get('embedding_model_recommendation', 'Not specified')}")
        print(f"Top-K: {parameters.get('top_k', 'Not specified')}")
        print(f"Retrieval Enhancements: {parameters.get('retrieval_enhancements', 'Not specified')}")
        print(f"Chunking Strategy: {parameters.get('chunking_strategy_recommendation', 'Not specified')}")
    else:  # pretty
        print("\nOptimized RAG Parameters:")
        print(json.dumps(parameters, indent=2))
        
        # Print practical interpretation
        print("\nPractical Interpretation:")
        chunk_size = parse_chunk_size(parameters.get('chunk_size_tokens', '~1000'))
        overlap = parse_overlap(parameters.get('overlap_tokens', '~100'))
        top_k = parse_top_k(parameters.get('top_k', '5'))
        embedding_model = map_embedding_recommendation(parameters.get('embedding_model_recommendation', ''))
        reranker = map_reranker_recommendation(parameters.get('retrieval_enhancements', ''))
        temperature = parse_temperature(parameters.get('generation_settings', ''))
        
        print(f"- Chunk Size: {chunk_size} tokens")
        print(f"- Chunk Overlap: {overlap} tokens")
        print(f"- Top-K: {top_k} chunks")
        print(f"- Embedding Model: {embedding_model}")
        print(f"- Reranker: {'Yes - ' + reranker if reranker else 'No'}")
        print(f"- Temperature: {temperature}")

def interactive_mode() -> Dict[str, Any]:
    """
    Run the optimizer in interactive mode, prompting the user for input.
    
    Returns:
        Dictionary of optimized RAG parameters
    """
    print("\nRAG Parameter Optimizer - Interactive Mode")
    print("------------------------------------------")
    
    # Prompt for use case
    print("\nAvailable Use Cases:")
    for i, use_case in enumerate(USE_CASES, 1):
        print(f"  {i}. {use_case}")
    
    while True:
        try:
            use_case_idx = int(input("\nSelect a use case (1-9): ")) - 1
            if 0 <= use_case_idx < len(USE_CASES):
                use_case = USE_CASES[use_case_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Prompt for document type
    print("\nDocument Types:")
    for i, doc_type in enumerate(DOCUMENT_TYPES, 1):
        print(f"  {i}. {doc_type}")
    
    while True:
        try:
            doc_type_idx = int(input("\nSelect a document type (1-5): ")) - 1
            if 0 <= doc_type_idx < len(DOCUMENT_TYPES):
                document_type = DOCUMENT_TYPES[doc_type_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Prompt for performance priority
    print("\nPerformance Priorities:")
    for i, priority in enumerate(PERFORMANCE_PRIORITIES, 1):
        print(f"  {i}. {priority}")
    
    while True:
        try:
            priority_idx = int(input("\nSelect a performance priority (1-4): ")) - 1
            if 0 <= priority_idx < len(PERFORMANCE_PRIORITIES):
                performance_priority = PERFORMANCE_PRIORITIES[priority_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Prompt for data size
    print("\nData Sizes:")
    for i, size in enumerate(DATA_SIZES, 1):
        print(f"  {i}. {size}")
    
    while True:
        try:
            size_idx = int(input("\nSelect a data size (1-3): ")) - 1
            if 0 <= size_idx < len(DATA_SIZES):
                data_size = DATA_SIZES[size_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"\nGetting optimized parameters for:")
    print(f"  Use Case: {use_case}")
    print(f"  Document Type: {document_type}")
    print(f"  Performance Priority: {performance_priority}")
    print(f"  Data Size: {data_size}")
    
    return get_optimized_parameters(use_case, document_type, performance_priority, data_size)

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="RAG Parameter Optimizer CLI")
    
    # Add arguments
    parser.add_argument("--use-case", choices=USE_CASES, help="The primary use case")
    parser.add_argument("--document-type", choices=DOCUMENT_TYPES, default="general", help="Type of documents")
    parser.add_argument("--performance", choices=PERFORMANCE_PRIORITIES, default="balanced", help="Optimization priority")
    parser.add_argument("--data-size", choices=DATA_SIZES, default="medium", help="Size of the data corpus")
    parser.add_argument("--output", help="Path to save the parameters as JSON")
    parser.add_argument("--format", choices=["pretty", "json", "compact"], default="pretty", help="Output format")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if interactive mode is selected
    if args.interactive:
        parameters = interactive_mode()
    else:
        # Check if use case is provided
        if not args.use_case:
            parser.error("--use-case is required unless --interactive is specified")
        
        # Get optimized parameters
        parameters = get_optimized_parameters(
            args.use_case,
            args.document_type,
            args.performance,
            args.data_size
        )
    
    # Print parameters
    print_parameters(parameters, args.format)
    
    # Save parameters to file if output is specified
    if args.output:
        save_parameters_to_file(parameters, args.output)

if __name__ == "__main__":
    main()
