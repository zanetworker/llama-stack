import os
import asyncio
import json
from termcolor import cprint
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool

from experiments.rag_parameter_optimizer_enhanced import get_rag_parameters
from experiments.rag_integration_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation
)

# Define the RAG parameter optimizer tool
@client_tool
def optimize_rag_parameters(
    use_case: str,
    document_type: str = "general",
    performance_priority: str = "balanced",
    data_size: str = "medium"
) -> dict:
    """
    Optimizes RAG parameters based on use case and other factors.
    
    :param use_case: The primary use case (e.g., "Knowledge Management", "Customer Support")
    :param document_type: Type of documents (e.g., "technical", "legal", "educational", "code")
    :param performance_priority: Optimization priority (e.g., "accuracy", "latency", "cost", "balanced")
    :param data_size: Size of the data corpus (e.g., "small", "medium", "large")
    :returns: Dictionary of optimized RAG parameters
    """
    # Get optimized parameters
    params = get_rag_parameters(use_case, document_type, performance_priority, data_size)
    return params

# Define the RAG pipeline configuration tool
@client_tool
def configure_rag_pipeline(parameters: dict) -> dict:
    """
    Configures a RAG pipeline based on the provided parameters.
    
    :param parameters: Dictionary of RAG parameters
    :returns: Dictionary containing the configured pipeline components
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

async def run_main():
    """Main function to run the RAG parameter optimizer agent."""
    # Get the model ID from environment variables
    model_id = os.environ.get("INFERENCE_MODEL", "meta-llama/Llama-3-8b-instruct")
    
    # Create a LlamaStackClient
    client = LlamaStackClient(
        base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
    )
    
    # Check available providers
    available_providers = client.providers.list()
    cprint(f"Available providers: {[p.provider_id for p in available_providers]}", "yellow")
    
    # Define the tool definitions
    rag_optimizer_tool = {
        "tool_name": "optimize_rag_parameters",
        "description": "Optimizes RAG parameters based on use case and other factors.",
        "parameters": {
            "use_case": {
                "param_type": "string",
                "description": "The primary use case (e.g., 'Knowledge Management', 'Customer Support')"
            },
            "document_type": {
                "param_type": "string",
                "description": "Type of documents (e.g., 'technical', 'legal', 'educational', 'code')"
            },
            "performance_priority": {
                "param_type": "string",
                "description": "Optimization priority (e.g., 'accuracy', 'latency', 'cost', 'balanced')"
            },
            "data_size": {
                "param_type": "string",
                "description": "Size of the data corpus (e.g., 'small', 'medium', 'large')"
            }
        }
    }
    
    rag_pipeline_tool = {
        "tool_name": "configure_rag_pipeline",
        "description": "Configures a RAG pipeline based on the provided parameters.",
        "parameters": {
            "parameters": {
                "param_type": "object",
                "description": "Dictionary of RAG parameters"
            }
        }
    }
    
    # Create an agent with the RAG parameter optimizer tools
    agent_config = AgentConfig(
        model=model_id,
        instructions="""You are a RAG optimization assistant. Help users configure optimal RAG parameters for their specific use case.
When a user describes their RAG needs:
1. Identify their use case (Knowledge Management, Customer Support, etc.)
2. Determine document type, performance priorities, and data size
3. Use the optimize_rag_parameters tool to get recommendations
4. Explain the recommendations and how they address the user's needs
5. If requested, use the configure_rag_pipeline tool to show how these parameters would be applied in a RAG pipeline
6. Offer to customize parameters if needed""",
        toolgroups=[],
        input_shields=[],
        output_shields=[],
        enable_session_persistence=True,
        tool_choice="auto",
        tool_prompt_format="python_list",
        max_tool_calls=3,
    )
    
    # Create the agent
    agent = Agent(
        client,
        model=model_id,
        instructions="""You are a RAG optimization assistant. Help users configure optimal RAG parameters for their specific use case.
When a user describes their RAG needs:
1. Identify their use case (Knowledge Management, Customer Support, etc.)
2. Determine document type, performance priorities, and data size
3. Use the optimize_rag_parameters tool to get recommendations
4. Explain the recommendations and how they address the user's needs
5. If requested, use the configure_rag_pipeline tool to show how these parameters would be applied in a RAG pipeline
6. Offer to customize parameters if needed""",
        tool_config={
            "tool_choice": "auto",
            "tool_prompt_format": "python_list"
        },
        tools=[rag_optimizer_tool, rag_pipeline_tool],
        input_shields=[],
        output_shields=[],
        enable_session_persistence=True,
        max_infer_iters=3
    )
    cprint("Successfully created agent", "green")
    
    # Create a session
    session_id = agent.create_session("rag-optimizer-session")
    cprint(f"Created session: {session_id}", "green")
    
    # Example prompts to test the agent
    test_prompts = [
        "I need help optimizing RAG for a knowledge management system with technical documentation. We prioritize accuracy over speed.",
        "Can you help me configure RAG for customer support? We have a small dataset and need fast responses.",
        "I'm building a code assistance tool. What RAG parameters should I use?",
        "We're developing a healthcare application with a large dataset of medical records. What RAG parameters would you recommend?"
    ]
    
    # Run the agent with each test prompt
    for i, prompt in enumerate(test_prompts):
        cprint(f"\n--- Test {i+1}: {prompt} ---", "cyan")
        
        try:
            # Create a turn with error handling
            response = agent.create_turn(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                session_id=session_id,
            )
            
            # Use try-except for each log to handle potential None values
            for log in EventLogger().log(response):
                try:
                    if log is not None:
                        log.print()
                    else:
                        cprint("Received None log entry", "yellow")
                except Exception as e:
                    cprint(f"Error processing log: {e}", "red")
        except Exception as e:
            cprint(f"Error creating turn: {e}", "red")
            import traceback
            traceback.print_exc()

def main():
    """Entry point for the script."""
    asyncio.run(run_main())

if __name__ == "__main__":
    main()
