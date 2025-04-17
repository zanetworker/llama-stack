#!/usr/bin/env python3
"""
RAG Parameter Visualizer

This script provides visualizations of how RAG parameters change based on
different use cases, document types, performance priorities, and data sizes.
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k
)

# Available options
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

def get_numeric_parameters(
    use_case: str,
    document_type: str = "general",
    performance_priority: str = "balanced",
    data_size: str = "medium"
) -> Dict[str, int]:
    """
    Get numeric parameters for visualization.
    
    Args:
        use_case: The primary use case
        document_type: Type of documents
        performance_priority: Optimization priority
        data_size: Size of the data corpus
        
    Returns:
        Dictionary of numeric parameters
    """
    params = get_rag_parameters(use_case, document_type, performance_priority, data_size)
    
    return {
        'chunk_size': parse_chunk_size(params.get('chunk_size_tokens', '~1000')),
        'overlap': parse_overlap(params.get('overlap_tokens', '~100')),
        'top_k': parse_top_k(params.get('top_k', '5'))
    }

def plot_parameter_comparison(
    parameter: str,
    use_cases: List[str] = USE_CASES,
    document_type: str = "general",
    performance_priority: str = "balanced",
    data_size: str = "medium"
) -> None:
    """
    Plot a comparison of a parameter across different use cases.
    
    Args:
        parameter: The parameter to compare ('chunk_size', 'overlap', or 'top_k')
        use_cases: List of use cases to compare
        document_type: Type of documents
        performance_priority: Optimization priority
        data_size: Size of the data corpus
    """
    values = []
    
    for use_case in use_cases:
        params = get_numeric_parameters(use_case, document_type, performance_priority, data_size)
        values.append(params[parameter])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(use_cases, values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            str(int(height)),
            ha='center', va='bottom'
        )
    
    plt.title(f'{parameter.replace("_", " ").title()} by Use Case')
    plt.xlabel('Use Case')
    plt.ylabel(parameter.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{parameter}_by_use_case.png')
    plt.close()

def plot_parameter_heatmap(
    parameter: str,
    use_case: str = "Knowledge Management",
    document_types: List[str] = DOCUMENT_TYPES,
    performance_priorities: List[str] = PERFORMANCE_PRIORITIES
) -> None:
    """
    Plot a heatmap of a parameter across different document types and performance priorities.
    
    Args:
        parameter: The parameter to visualize ('chunk_size', 'overlap', or 'top_k')
        use_case: The use case to analyze
        document_types: List of document types to compare
        performance_priorities: List of performance priorities to compare
    """
    values = np.zeros((len(document_types), len(performance_priorities)))
    
    for i, doc_type in enumerate(document_types):
        for j, perf in enumerate(performance_priorities):
            params = get_numeric_parameters(use_case, doc_type, perf)
            values[i, j] = params[parameter]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(values, cmap='viridis')
    
    # Add value labels
    for i in range(len(document_types)):
        for j in range(len(performance_priorities)):
            plt.text(
                j, i, str(int(values[i, j])),
                ha='center', va='center',
                color='white' if values[i, j] > np.mean(values) else 'black'
            )
    
    plt.title(f'{parameter.replace("_", " ").title()} Heatmap for {use_case}')
    plt.xlabel('Performance Priority')
    plt.ylabel('Document Type')
    plt.xticks(np.arange(len(performance_priorities)), performance_priorities)
    plt.yticks(np.arange(len(document_types)), document_types)
    plt.colorbar(label=parameter.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(f'{parameter}_heatmap_{use_case.replace(" ", "_").lower()}.png')
    plt.close()

def plot_data_size_impact(
    parameter: str,
    use_cases: List[str] = USE_CASES[:4],  # Limit to a few use cases for clarity
    document_type: str = "general",
    performance_priority: str = "balanced"
) -> None:
    """
    Plot how data size affects a parameter across different use cases.
    
    Args:
        parameter: The parameter to analyze ('chunk_size', 'overlap', or 'top_k')
        use_cases: List of use cases to compare
        document_type: Type of documents
        performance_priority: Optimization priority
    """
    small_values = []
    medium_values = []
    large_values = []
    
    for use_case in use_cases:
        small_params = get_numeric_parameters(use_case, document_type, performance_priority, "small")
        medium_params = get_numeric_parameters(use_case, document_type, performance_priority, "medium")
        large_params = get_numeric_parameters(use_case, document_type, performance_priority, "large")
        
        small_values.append(small_params[parameter])
        medium_values.append(medium_params[parameter])
        large_values.append(large_params[parameter])
    
    x = np.arange(len(use_cases))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, small_values, width, label='Small Data')
    plt.bar(x, medium_values, width, label='Medium Data')
    plt.bar(x + width, large_values, width, label='Large Data')
    
    plt.title(f'Impact of Data Size on {parameter.replace("_", " ").title()}')
    plt.xlabel('Use Case')
    plt.ylabel(parameter.replace('_', ' ').title())
    plt.xticks(x, use_cases, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'data_size_impact_{parameter}.png')
    plt.close()

def main():
    """Main function to generate visualizations."""
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    os.chdir('visualizations')
    
    print("Generating parameter comparisons across use cases...")
    plot_parameter_comparison('chunk_size')
    plot_parameter_comparison('overlap')
    plot_parameter_comparison('top_k')
    
    print("Generating parameter heatmaps...")
    plot_parameter_heatmap('chunk_size', "Knowledge Management")
    plot_parameter_heatmap('top_k', "Code Assistance")
    
    print("Generating data size impact visualizations...")
    plot_data_size_impact('chunk_size')
    plot_data_size_impact('top_k')
    
    print("Visualizations generated successfully!")
    print(f"Output saved to: {os.path.abspath('.')}")

if __name__ == "__main__":
    main()
