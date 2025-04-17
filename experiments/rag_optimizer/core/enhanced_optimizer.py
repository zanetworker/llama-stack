import json
import re
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.base_optimizer import get_rag_parameters as get_base_rag_parameters

def adjust_for_document_type(params, document_type):
    """
    Adjust RAG parameters based on document type.
    
    Args:
        params: Base parameters dictionary
        document_type: Type of documents (technical, legal, educational, code)
        
    Returns:
        Adjusted parameters dictionary
    """
    adjusted = params.copy()
    
    if document_type == "technical":
        # Technical documents often benefit from smaller chunks for precision
        if 'chunk_size_tokens' in adjusted:
            adjusted['chunk_size_tokens'] = '~1000'  # Smaller chunks
        # Technical documents often need specialized embeddings
        if 'embedding_model_recommendation' in adjusted:
            if 'fine-tuned' not in adjusted['embedding_model_recommendation'].lower():
                adjusted['embedding_model_recommendation'] = 'Technical domain-specific embeddings (e.g., SPECTER for scientific papers)'
    
    elif document_type == "legal":
        # Legal documents often benefit from larger chunks and higher overlap
        if 'chunk_size_tokens' in adjusted:
            adjusted['chunk_size_tokens'] = '~2500'  # Larger chunks
        if 'overlap_tokens' in adjusted:
            adjusted['overlap_tokens'] = '~300'  # Higher overlap
        # Legal documents often need specialized embeddings
        if 'embedding_model_recommendation' in adjusted:
            if 'fine-tuned' not in adjusted['embedding_model_recommendation'].lower():
                adjusted['embedding_model_recommendation'] = 'Legal domain-specific embeddings'
    
    elif document_type == "educational":
        # Educational content benefits from moderate chunks
        if 'chunk_size_tokens' in adjusted:
            adjusted['chunk_size_tokens'] = '~1500'  # Moderate chunks
        # Educational content often benefits from higher top-k for broader coverage
        if 'top_k' in adjusted:
            if isinstance(adjusted['top_k'], (int, float)):
                adjusted['top_k'] = max(adjusted['top_k'], 7)
            else:
                adjusted['top_k'] = '7-10'  # Higher top-k
    
    elif document_type == "code":
        # Code benefits from syntax-aware chunking
        if 'chunking_strategy_recommendation' in adjusted:
            adjusted['chunking_strategy_recommendation'] = 'Syntax-aware chunking at function/class level'
        # Code needs specialized embeddings
        if 'embedding_model_recommendation' in adjusted:
            adjusted['embedding_model_recommendation'] = 'Code-specific embeddings (e.g., CodeBERT)'
    
    return adjusted

def adjust_for_performance(params, performance_priority):
    """
    Adjust RAG parameters based on performance priority.
    
    Args:
        params: Base parameters dictionary
        performance_priority: Optimization priority (accuracy, latency, cost, balanced)
        
    Returns:
        Adjusted parameters dictionary
    """
    adjusted = params.copy()
    
    if performance_priority == "accuracy":
        # Higher quality embeddings
        if 'embedding_model_recommendation' in adjusted:
            if 'fine-tuned' not in adjusted['embedding_model_recommendation'].lower():
                adjusted['embedding_model_recommendation'] += ' (prefer larger, more accurate models)'
        # More retrieval enhancements
        if 'retrieval_enhancements' in adjusted:
            if 'rerank' not in adjusted['retrieval_enhancements'].lower():
                adjusted['retrieval_enhancements'] += ', with reranking for improved relevance'
        # Lower temperature
        if 'generation_settings' in adjusted:
            if 'temperature' in adjusted['generation_settings'].lower():
                adjusted['generation_settings'] = re.sub(r'temperature \(e\.g\., \d+\.\d+\)', 'temperature (e.g., 0.3)', adjusted['generation_settings'])
            else:
                adjusted['generation_settings'] += ', lower temperature (e.g., 0.3) for factual responses'
    
    elif performance_priority == "latency":
        # Smaller models
        if 'embedding_model_recommendation' in adjusted:
            if 'fine-tuned' not in adjusted['embedding_model_recommendation'].lower():
                adjusted['embedding_model_recommendation'] += ' (prefer smaller, faster models)'
        # Fewer chunks retrieved
        if 'top_k' in adjusted:
            if isinstance(adjusted['top_k'], (int, float)):
                adjusted['top_k'] = min(adjusted['top_k'], 3)
            else:
                adjusted['top_k'] = '2-3'  # Lower top-k
        # Simpler retrieval
        if 'retrieval_enhancements' in adjusted:
            if 'hybrid' in adjusted['retrieval_enhancements'].lower() or 'rerank' in adjusted['retrieval_enhancements'].lower():
                adjusted['retrieval_enhancements'] = 'Basic vector search (optimized for speed)'
            else:
                adjusted['retrieval_enhancements'] += ' (optimized for speed)'
    
    elif performance_priority == "cost":
        # Smaller embedding models
        if 'embedding_model_recommendation' in adjusted:
            if 'fine-tuned' not in adjusted['embedding_model_recommendation'].lower():
                adjusted['embedding_model_recommendation'] += ' (prefer smaller, cost-effective models)'
        # Efficient chunking
        if 'chunking_strategy_recommendation' in adjusted:
            adjusted['chunking_strategy_recommendation'] += ' with cost-optimized parameters'
        # Fewer chunks retrieved
        if 'top_k' in adjusted:
            if isinstance(adjusted['top_k'], (int, float)):
                adjusted['top_k'] = min(adjusted['top_k'], 3)
            else:
                adjusted['top_k'] = '2-3'  # Lower top-k
    
    return adjusted

def adjust_for_data_size(params, data_size):
    """
    Adjust RAG parameters based on data size.
    
    Args:
        params: Base parameters dictionary
        data_size: Size of the data corpus (small, medium, large)
        
    Returns:
        Adjusted parameters dictionary
    """
    adjusted = params.copy()
    
    if data_size == "small":
        # Lower chunk size
        if 'chunk_size_tokens' in adjusted:
            adjusted['chunk_size_tokens'] = '~800'  # Smaller chunks
        # Lower top-k
        if 'top_k' in adjusted:
            if isinstance(adjusted['top_k'], (int, float)):
                adjusted['top_k'] = min(adjusted['top_k'], 3)
            else:
                adjusted['top_k'] = '2-3'  # Lower top-k
    
    elif data_size == "large":
        # Higher chunk size
        if 'chunk_size_tokens' in adjusted:
            adjusted['chunk_size_tokens'] = '~2000'  # Larger chunks
        # Higher top-k
        if 'top_k' in adjusted:
            if isinstance(adjusted['top_k'], (int, float)):
                adjusted['top_k'] = max(adjusted['top_k'], 7)
            else:
                adjusted['top_k'] = '7-10'  # Higher top-k
        # More sophisticated chunking
        if 'chunking_strategy_recommendation' in adjusted:
            if 'hierarchical' not in adjusted['chunking_strategy_recommendation'].lower():
                adjusted['chunking_strategy_recommendation'] += ' or Hierarchical for large datasets'
    
    return adjusted

def get_rag_parameters(use_case, document_type="general", performance_priority="balanced", data_size="medium"):
    """
    Get optimized RAG parameters based on use case and additional factors.
    
    Args:
        use_case: The primary use case (e.g., "Knowledge Management", "Customer Support")
        document_type: Type of documents (e.g., "technical", "legal", "educational", "code")
        performance_priority: Optimization priority (e.g., "accuracy", "latency", "cost", "balanced")
        data_size: Size of the data corpus (e.g., "small", "medium", "large")
        
    Returns:
        Dictionary of optimized RAG parameters
    """
    # Get base parameters for the use case
    params = get_base_rag_parameters(use_case)
    
    # Apply adjustments based on document type
    if document_type != "general":
        params = adjust_for_document_type(params, document_type)
    
    # Apply adjustments based on performance priority
    if performance_priority != "balanced":
        params = adjust_for_performance(params, performance_priority)
    
    # Apply adjustments based on data size
    if data_size != "medium":
        params = adjust_for_data_size(params, data_size)
    
    # Add metadata about the adjustments
    params['adjustments_applied'] = {
        'document_type': document_type,
        'performance_priority': performance_priority,
        'data_size': data_size
    }
    
    return params

# Example Usage
if __name__ == "__main__":
    print("Enhanced RAG Parameter Optimizer Examples:\n")

    test_cases = [
        {
            "use_case": "Knowledge Management",
            "document_type": "technical",
            "performance_priority": "accuracy",
            "data_size": "large"
        },
        {
            "use_case": "Customer Support",
            "document_type": "general",
            "performance_priority": "latency",
            "data_size": "medium"
        },
        {
            "use_case": "Code Assistance",
            "document_type": "code",
            "performance_priority": "balanced",
            "data_size": "small"
        }
    ]

    for case in test_cases:
        print(f"--- Parameters for Use Case: {case['use_case']} ---")
        print(f"Document Type: {case['document_type']}")
        print(f"Performance Priority: {case['performance_priority']}")
        print(f"Data Size: {case['data_size']}")
        
        parameters = get_rag_parameters(
            case['use_case'],
            case['document_type'],
            case['performance_priority'],
            case['data_size']
        )
        
        # Pretty print the dictionary
        print(json.dumps(parameters, indent=4))
        print("-" * 80 + "\n")
