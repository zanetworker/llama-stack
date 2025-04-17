import json

# Base default parameters for unknown or unspecified use cases
BASE_DEFAULTS = {
    'use_case': 'Default',
    'chunk_size_tokens': '~1800',
    'overlap_tokens': '~200',
    'embedding_model_recommendation': 'General purpose (e.g., gte-large-en-v1.5)',
    'top_k': 5,
    'metadata_usage': 'Consider document source, date',
    'retrieval_enhancements': 'Basic vector search',
    'generation_settings': 'Moderate temperature (e.g., 0.5)',
    'prompting_technique': 'Standard instruction prompting',
    'output_guardrails': 'Consider basic safety filters',
    'chunking_strategy_recommendation': 'Recursive or Semantic',
    'notes': 'Using base defaults. Fine-tuning recommended for specific needs.'
}

# Detailed parameter recommendations based on use case
USE_CASE_PARAMS = {
    "Knowledge Management": {
        'use_case': 'Knowledge Management',
        'chunk_size_tokens': '1500-2000',
        'overlap_tokens': '~200',
        'embedding_model_recommendation': 'Domain-specific fine-tuned on enterprise data',
        'top_k': '3-5',
        'metadata_usage': 'Document type, author tags',
        'retrieval_enhancements': 'Basic vector search, consider reranking',
        'generation_settings': 'None specified',
        'prompting_technique': 'Standard',
        'output_guardrails': 'None specified',
        'chunking_strategy_recommendation': 'Recursive/Semantic',
    },
    "Customer Support": {
        'use_case': 'Customer Support',
        'chunk_size_tokens': 'Not specified, consider moderate (~1000-1500)', # Inference
        'overlap_tokens': 'Not specified, consider moderate (~100-150)', # Inference
        'embedding_model_recommendation': 'Fine-tune on support tickets/FAQs', # Inference
        'top_k': 'Not specified, consider 3-5', # Inference
        'metadata_usage': 'Ticket ID, customer segment, product area', # Inference
        'retrieval_enhancements': 'Query expansion/rewriting, Reranking (RRF)',
        'generation_settings': 'Consider lower temperature for factual answers', # Inference
        'prompting_technique': 'Standard, potentially few-shot for common issues', # Inference
        'output_guardrails': 'Fact-checking filters, alignment with service guidelines',
        'chunking_strategy_recommendation': 'Semantic or Sentence-based for FAQs', # Inference
    },
    "Healthcare Applications": {
        'use_case': 'Healthcare Applications',
        'chunk_size_tokens': '~512',
        'overlap_tokens': 'Not specified, consider small overlap (e.g., 50)', # Inference
        'embedding_model_recommendation': 'Fine-tune on medical datasets (e.g., PubMedBERT or fine-tuned general)',
        'top_k': 'Not specified, consider moderate k (e.g., 5-7) for context', # Inference
        'metadata_usage': 'Patient context (anonymized), document type (study, notes), date', # Inference
        'retrieval_enhancements': 'Hierarchical/Multi-hop retrieval',
        'generation_settings': 'Higher precision, potentially lower temperature', # Inference
        'prompting_technique': 'Consider CoT for complex reasoning', # Inference
        'output_guardrails': 'Strong fact-checking against medical knowledge bases', # Inference
        'chunking_strategy_recommendation': 'Smaller chunks for precision, potentially sentence-level',
    },
    "Education": {
        'use_case': 'Education',
        'chunk_size_tokens': 'Not specified, consider moderate (1000-2000)', # Inference
        'overlap_tokens': 'Not specified, consider moderate (~150-200)', # Inference
        'embedding_model_recommendation': 'General purpose or fine-tuned on subject matter',
        'top_k': 10, # Higher k for broader coverage
        'metadata_usage': 'Subject, topic, difficulty level, source document', # Inference
        'retrieval_enhancements': 'Basic vector search, potentially reranking for relevance',
        'generation_settings': 'LLM optimized for summarization/explanation',
        'prompting_technique': 'Chain-of-Thought (CoT) for explanations',
        'output_guardrails': 'Ensure age-appropriateness, factual accuracy', # Inference
        'chunking_strategy_recommendation': 'Recursive/Semantic based on textbook structure', # Inference
    },
    "Code Assistance": {
        'use_case': 'Code Assistance',
        'chunk_size_tokens': 'Not specified, depends on function/class size', # Inference
        'overlap_tokens': 'Consider overlap to capture context across blocks', # Inference
        'embedding_model_recommendation': 'Code-specific (e.g., CodeBERT, UniXCoder) or fine-tuned',
        'top_k': 'Not specified, consider 5-10 snippets', # Inference
        'metadata_usage': 'File type, function/class name, library dependencies',
        'retrieval_enhancements': 'Vector search, potentially hybrid with keyword for specific syntax', # Inference based on prior discussion
        'generation_settings': 'LLM fine-tuned for code generation/explanation', # Inference
        'prompting_technique': 'Clear instructions, provide context (e.g., surrounding code)', # Inference
        'output_guardrails': 'Syntax checking, basic security checks', # Inference
        'chunking_strategy_recommendation': 'Recursive with syntax-aware splitting (function/class level)',
    },
    "Sales Automation": {
        'use_case': 'Sales Automation',
        'chunk_size_tokens': 'Not specified, consider moderate (1000-1500)', # Inference
        'overlap_tokens': 'Not specified, consider moderate (~100-150)', # Inference
        'embedding_model_recommendation': 'Personalized embeddings fine-tuned on customer profiles/sales data',
        'top_k': 'Not specified, consider 3-5 relevant items', # Inference
        'metadata_usage': 'Product categories, customer demographics, lead source',
        'retrieval_enhancements': 'Vector search focused on personalization',
        'generation_settings': 'Lower temperature (0.2-0.4) for deterministic/professional responses',
        'prompting_technique': 'Standard, potentially persona-driven', # Inference
        'output_guardrails': 'Ensure brand voice consistency, accuracy of product info', # Inference
        'chunking_strategy_recommendation': 'Semantic chunking of product descriptions/customer interactions', # Inference
    },
    "Marketing": {
        'use_case': 'Marketing',
        'chunk_size_tokens': 'Not specified, consider moderate (1500-2000) for case studies', # Inference
        'overlap_tokens': 'Not specified, consider moderate (~150-200)', # Inference
        'embedding_model_recommendation': 'Fine-tune on marketing data (case studies, campaign data)',
        'top_k': 'Not specified, consider 5-7 for summarization/ideation', # Inference
        'metadata_usage': 'Campaign name, target audience, content type (blog, ad copy)', # Inference
        'retrieval_enhancements': 'Vector search, potentially reranking for relevance to campaign goals',
        'generation_settings': 'LLM optimized for summarization/creative generation', # Inference based on description
        'prompting_technique': 'Standard, potentially creative prompting techniques', # Inference
        'output_guardrails': 'Ensure factual accuracy in summaries, brand consistency', # Inference based on description
        'chunking_strategy_recommendation': 'Semantic chunking of marketing materials', # Inference
    },
    "Threat Analysis": {
        'use_case': 'Threat Analysis',
        'chunk_size_tokens': '~2500', # Larger chunks for dense technical details
        'overlap_tokens': '~250',
        'embedding_model_recommendation': 'Fine-tune on cybersecurity reports/threat intel', # Inference
        'top_k': 'Not specified, consider higher k (e.g., 10-15) for comprehensive analysis', # Inference
        'metadata_usage': 'Threat actor, vulnerability ID (CVE), report date, source', # Inference
        'retrieval_enhancements': 'Hybrid reranking (Sparse/Dense, e.g., BM25 + embeddings)',
        'generation_settings': 'Focus on factual extraction and summarization', # Inference
        'prompting_technique': 'Standard, potentially CoT for linking indicators', # Inference
        'output_guardrails': 'High emphasis on factual accuracy, avoid speculation', # Inference
        'chunking_strategy_recommendation': 'Larger chunks, potentially recursive based on report structure',
    },
    "Gaming": {
        'use_case': 'Gaming',
        'chunk_size_tokens': 'Not specified, depends on narrative element size (dialogue, lore entry)', # Inference
        'overlap_tokens': 'Consider overlap for smooth transitions', # Inference
        'embedding_model_recommendation': 'Fine-tune on game lore, character dialogues', # Inference
        'top_k': 3, # Lower k for focused storytelling
        'metadata_usage': 'Character name, location, quest ID, dialogue turn', # Inference
        'retrieval_enhancements': 'Vector search, potentially hierarchical for story context', # Inference based on description
        'generation_settings': 'LLM tuned for creative writing/dialogue', # Inference
        'prompting_technique': 'Persona-driven, context-aware prompting', # Inference
        'output_guardrails': 'Maintain character consistency, plot coherence', # Inference
        'chunking_strategy_recommendation': 'Dynamic/Hierarchical chunk selection for context',
    }
    # Add 'Specialized Applications' if specific parameters become known
}

def get_rag_parameters(use_case: str) -> dict:
    """
    Retrieves recommended RAG parameters based on the specified use case.

    Args:
        use_case: The name of the use case (e.g., "Customer Support", "Code Assistance").

    Returns:
        A dictionary containing recommended parameters, or base defaults if
        the use case is not found.
    """
    # Normalize input slightly (optional, but can help with minor variations)
    normalized_use_case = use_case.strip().title() # Capitalize first letter of each word

    params = USE_CASE_PARAMS.get(normalized_use_case)

    if params:
        return params.copy() # Return a copy to prevent modification of the original dict
    else:
        # Return base defaults if use case not found, adding a note
        defaults = BASE_DEFAULTS.copy()
        defaults['notes'] = f"Specific parameters for use case '{use_case}' not found. Using base defaults. Fine-tuning recommended."
        defaults['requested_use_case'] = use_case # Keep track of what was asked for
        return defaults

# Example Usage
if __name__ == "__main__":
    print("RAG Parameter Optimizer Examples:\n")

    test_cases = [
        "Knowledge Management",
        "Customer Support",
        "Healthcare Applications",
        "Education",
        "Code Assistance",
        "Sales Automation",
        "Marketing",
        "Threat Analysis",
        "Gaming",
        "Financial Analysis", # Example of an unknown use case
        "   code assistance   " # Example with different casing/spacing
    ]

    for case in test_cases:
        print(f"--- Parameters for Use Case: {case} ---")
        parameters = get_rag_parameters(case)
        # Pretty print the dictionary
        print(json.dumps(parameters, indent=4))
        print("-" * (len(f"--- Parameters for Use Case: {case} ---") + 1) + "\n")
