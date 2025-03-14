"""
Demo script for the Agent Orchestration System

This script demonstrates key capabilities of the system with a sample workflow.
"""

import os
import asyncio
import logging
from termcolor import cprint

from llama_stack_client import LlamaStackClient
from specialized_agents import WebAgent, CodeAgent, FileAgent, RAGAgent, create_agent_by_profile

from orchestrator import OrchestratorAgent, SelfPlayCustomization, Task, Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_specialized_agents():
    """Demonstrate specialized agent capabilities"""
    # Get config from environment
    port = os.environ.get('LLAMA_STACK_PORT', '8080')
    model_id = os.environ.get('INFERENCE_MODEL', 'llama3')
    
    # Initialize client
    client = LlamaStackClient(
        base_url=f"http://localhost:{port}"
    )
    
    cprint("\n==== Specialized Agents Demo ====", "blue")
    
    # Create a web agent
    web_agent = WebAgent(client, model_id, "demo_web_agent")
    agent, session_id = web_agent.create_instance()
    
    # Use the web agent
    query = "What is the current weather in San Francisco?"
    cprint(f"\nUser> {query}", "green")
    
    try:
        response = agent.create_turn(
            messages=[{
                "role": "user",
                "content": query
            }],
            session_id=session_id
        )
        
        cprint("\nWeb Agent response:", "yellow")
        for chunk in response:
            if chunk and hasattr(chunk, 'content') and chunk.content:
                cprint(chunk.content, "white", end="")
        print()
    except Exception as e:
        logger.error(f"Error with web agent: {e}")


async def demo_task_planning():
    """Demonstrate task planning capabilities"""
    # Get config from environment
    port = os.environ.get('LLAMA_STACK_PORT', '8080')
    model_id = os.environ.get('INFERENCE_MODEL', 'llama3')
    
    # Initialize client
    client = LlamaStackClient(
        base_url=f"http://localhost:{port}"
    )
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(client, model_id)
    
    cprint("\n==== Task Planning Demo ====", "blue")
    
    # Complex task for planning
    complex_task = "Create a Python script that fetches stock data for Apple, analyzes the trends for the last 30 days, and generates a visualization and execute it."
    
    cprint(f"\nComplex task: {complex_task}", "green")
    tasks = orchestrator.parse_task(complex_task)
    
    cprint("\nTask breakdown:", "yellow")
    for task in tasks:
        cprint(f"- Task {task.task_id}: {task.description}", "white")
        cprint(f"  Agent: {task.agent_id}", "cyan")


async def demo_self_play():
    """Demonstrate self-play customization"""
    # Get config from environment
    port = os.environ.get('LLAMA_STACK_PORT', '8080')
    model_id = os.environ.get('INFERENCE_MODEL', 'llama3')
    
    # Initialize client
    client = LlamaStackClient(
        base_url=f"http://localhost:{port}"
    )
    
    # Initialize self-play
    self_play = SelfPlayCustomization(client, model_id)
    
    cprint("\n==== Self-Play Customization Demo ====", "blue")
    
    # Requirements for a custom agent
    requirements = """
    Create an agent that specializes in data visualization and analysis.
    It should be able to:
    1. Process CSV and JSON data files
    2. Generate various chart types (bar, line, scatter, etc.)
    3. Provide statistical analysis of data
    4. Generate insights from the visualizations
    """
    
    cprint(f"\nRequirements: {requirements}", "green")
    
    # Generate agent profile
    profile = self_play.generate_agent_profile(requirements)
    
    cprint("\nGenerated agent profile:", "yellow")
    cprint(f"Agent ID: {profile.get('agent_id', 'Unknown')}", "white")
    cprint(f"Agent Type: {profile.get('agent_type', 'Unknown')}", "white")
    cprint(f"Instructions: {profile.get('instructions', 'None')[:100]}...", "white")
    cprint(f"Required Tools: {profile.get('required_tools', [])}", "white")
    
    # Map tools and create agent
    toolgroups = self_play.map_tools_to_toolgroups(profile.get("required_tools", []))
    
    cprint("\nMapped tools:", "yellow")
    for tool in toolgroups:
        cprint(f"- {tool.get('name', 'Unknown')}", "white")


async def demo_full_workflow():
    """Demonstrate a complete workflow through the system"""
    # Get config from environment
    port = os.environ.get('LLAMA_STACK_PORT', '8080')
    model_id = os.environ.get('INFERENCE_MODEL', 'llama3')
    
    # Initialize client
    client = LlamaStackClient(
        base_url=f"http://localhost:{port}"
    )
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(client, model_id)
    
    # Register specialized agents
    orchestrator.register_specialized_agent(
        agent_id="web_agent",
        agent_type="WebAgent",
        instructions="""You are a specialized web agent. Your task is to:
        1. Search the web for information
        2. Extract relevant data from websites
        3. Present the information clearly and concisely
        """,
        toolgroups=[{
            "name": "builtin::websearch",
            "args": {}
        }]
    )
    
    orchestrator.register_specialized_agent(
        agent_id="code_agent",
        agent_type="CodeAgent",
        instructions="""You are a specialized code agent. Your task is to:
        1. Write, analyze, or debug code
        2. Explain programming concepts
        3. Provide code-related guidance
        """,
        toolgroups=[{
            "name": "builtin::code_interpreter",
            "args": {}
        }]
    )
    
    cprint("\n==== Full Workflow Demo ====", "blue")
    
    # User request
    user_request = "Find information about Python's pandas library and create a sample script to analyze stock data"
    
    cprint(f"\nUser request: {user_request}", "green")
    
    # Process the request
    try:
        response = await orchestrator.process_request(user_request)
        
        cprint("\nFinal response:", "yellow")
        cprint(response, "white")
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all demos"""
    # Check which demos to run based on environment variables or defaults
    run_specialized = os.environ.get('DEMO_SPECIALIZED', 'true').lower() == 'true'
    run_task_planning = os.environ.get('DEMO_TASK_PLANNING', 'true').lower() == 'true'
    run_self_play = os.environ.get('DEMO_SELF_PLAY', 'true').lower() == 'true'
    run_workflow = os.environ.get('DEMO_WORKFLOW', 'true').lower() == 'true'
    
    # Run selected demos
    if run_specialized:
        await demo_specialized_agents()
    
    if run_task_planning:
        await demo_task_planning()
    
    if run_self_play:
        await demo_self_play()
    
    if run_workflow:
        await demo_full_workflow()


if __name__ == "__main__":
    asyncio.run(main()) 